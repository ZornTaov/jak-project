"""
Rebuild the eyes of merc models exported by the decompiler's glTF exporter, in Blender (4.2+).

In the game, eye materials use a placeholder texture (programmer_eye_left / programmer_eye_right) that is
replaced every frame by a texture composited by eye.gc (render-eyes-32/64) and EyeRenderer.cpp:
  1. fill with the iris texture's top-left texel (the sclera color)
  2. iris sprite, centered at (0.5 + x, 0.5 + y), iris-scale wide/tall, opaque
  3. pupil sprite, same center, pupil-scale wide/tall, alpha blended (alpha * 2)
  4. lid sprite, full width, top at (lid - 1), lid-scale tall, opaque. lid 0 = open (lid above the eye), 1 = closed
in eye texture units with y pointing down. The iris/pupil/lid textures come from the model's merc-eye-ctrl.

This script builds that as nodes in place of the placeholder texture. The eye parameters are custom
properties on the object ("jak_eye_l_x", "jak_eye_r_lid", ...), so they can be keyframed.
The eye textures are found by reading the model's art group (decompiler_out/<game>/raw_obj/<model>-ag.go) and
loading the PNGs from decompiler_out/<game>/textures/. Use EYE_TEXTURES to override.

Run jak_merc_materials.py first: the eye's envmap mask is rewired to the composited eye alpha.

Usage:
  - In Blender: set GAME_OUT_DIR, select the objects (or nothing, for all), run this script.
  - Command line: blender -b --python jak_eyes.py -- model.glb [out.blend]
    (imports the glb and runs jak_merc_materials.py first; GAME_OUT_DIR is found from the glb path)
"""

import re
import struct
import sys
from pathlib import Path

import bpy

# decompiler_out/<game>, e.g. r"G:\Ports\jak-project\decompiler_out\jak3". Found from the glb path on the
# command line, or from the .blend's path if it's saved somewhere inside it.
GAME_OUT_DIR = None

# Override the textures per art group, e.g. {"daxter-highres": ("autoeye-iris", "autoeye-pupil", "daxter-eyelid")}.
EYE_TEXTURES = {}

# Defaults for the eye custom properties: the game's values when a model has no eye animation (eye.gc).
EYE_DEFAULTS = {"x": 0.0, "y": 0.0, "lid": 0.0, "iris_scale": 0.55, "pupil_scale": 0.45, "lid_scale": 1.0}

EYE_LAYOUT_GROUP = "JakEyeLayout"
PLACEHOLDERS = {"programmer_eye_left": "l", "programmer_eye_right": "r"}
# merc-eye-ctrl shader order
SHADER_NAMES = ("iris", "pupil", "lid")
# adgif register addresses: tex0, tex1, miptbp1, clamp, alpha
ADGIF_ADDRS = (0x06, 0x14, 0x34, 0x08, 0x42)


def socket(sockets, identifier):
    for s in sockets:
        if s.identifier == identifier:
            return s
    raise KeyError(identifier)


def upstream(sock, node_type, depth=4):
    """Find the closest node of node_type feeding into sock."""
    for link in sock.links:
        node = link.from_node
        if node.type == node_type:
            return node
        if depth > 0:
            for inp in node.inputs:
                found = upstream(inp, node_type, depth - 1)
                if found:
                    return found
    return None


#######################
# Finding the textures
#######################


def find_game_out_dir(hint=None):
    if GAME_OUT_DIR:
        return Path(GAME_OUT_DIR)
    for path in (hint, bpy.data.filepath):
        if not path:
            continue
        p = Path(path).resolve()
        for parent in p.parents:
            if parent.parent.name == "decompiler_out":
                return parent
    return None


def read_texture_names(game_out, game):
    """(tpage id, texture index) -> texture name, from goal_src/<game>/engine/data."""
    data_dir = game_out.parent.parent / "goal_src" / game / "engine" / "data"
    tpages = {}
    for m in re.finditer(r"\(defconstant ([a-z0-9-]+) (\d+)\)", (data_dir / "tpages.gc").read_text()):
        tpages[m.group(1)] = int(m.group(2))
    names = {}
    for m in re.finditer(r"\(def-tex ([a-z0-9_-]+) ([a-z0-9-]+) (\d+)\)", (data_dir / "textures.gc").read_text()):
        if m.group(2) in tpages:
            names[(tpages[m.group(2)], int(m.group(3)))] = m.group(1)
    return names


def read_eye_shaders(ag_path, tex_names):
    """
    Find the merc-eye-ctrl in a raw art group: a 16 byte header (shader-count 6) followed by 6 adgif shaders,
    left iris/pupil/lid then right iris/pupil/lid. Returns 6 texture names (None if unknown, e.g. remapped).
    """
    data = ag_path.read_bytes()

    def is_adgif(off):
        return off >= 0 and all(
            off + 16 * i + 8 < len(data) and data[off + 16 * i + 8] == a for i, a in enumerate(ADGIF_ADDRS)
        )

    for off in range(16, len(data) - 6 * 80, 16):
        if data[off - 14] != 6 or is_adgif(off - 80):
            continue
        if all(is_adgif(off + 80 * i) for i in range(6)):
            result = []
            for i in range(6):
                tex = struct.unpack_from("<I", data, off + 80 * i + 16 + 8)[0]
                result.append(tex_names.get((tex >> 20, (tex >> 8) & 0xFFF)))
            return result
    return None


def guess_name(model, kind, textures_dir):
    """Fallback for textures in remapped tpages: <prefix>-eyelid, <prefix>-iris, ... for prefixes of the name."""
    suffixes = {"iris": ("iris", "eye"), "pupil": ("pupil",), "lid": ("eyelid", "eye-lid", "lid")}[kind]
    parts = model.split("-")
    for n in range(len(parts), 0, -1):
        for suffix in suffixes:
            name = "-".join(parts[:n]) + "-" + suffix
            if next(textures_dir.glob(f"*/{name}.png"), None):
                return name
    return None


def find_png(textures_dir, name, level=None):
    matches = sorted(textures_dir.glob(f"*/{name}.png"))
    if level:
        # prefer the level's own tpages
        matches.sort(key=lambda p: not p.parent.name.startswith(level))
    return matches[0] if matches else None


def find_eye_images(model, game_out, level=None):
    """Returns {"l": (iris, pupil, lid), "r": (...)} as bpy images, or None."""
    textures_dir = game_out / "textures"
    if model in EYE_TEXTURES:
        names = list(EYE_TEXTURES[model]) * 2
    else:
        ag_path = game_out / "raw_obj" / f"{model}-ag.go"
        if not ag_path.exists():
            print(f"  no art group {ag_path}, set EYE_TEXTURES")
            return None
        names = read_eye_shaders(ag_path, read_texture_names(game_out, game_out.name))
        if names is None:
            print(f"  no merc-eye-ctrl found in {ag_path.name}")
            return None
        names = [n or guess_name(model, SHADER_NAMES[i % 3], textures_dir) for i, n in enumerate(names)]

    images = []
    for name in names:
        path = find_png(textures_dir, name, level) if name else None
        if path is None:
            print(f"  can't find eye texture {name!r}, set EYE_TEXTURES")
            return None
        image = bpy.data.images.load(str(path), check_existing=True)
        # alpha is PS2 range (0x80 = 1.0), don't let it darken the color
        image.alpha_mode = "CHANNEL_PACKED"
        images.append(image)
    print(f"  eye textures: left {names[:3]}, right {names[3:]}")
    return {"l": images[:3], "r": images[3:]}


#######################
# Nodes
#######################


class Builder:
    """Small helper for building math node chains."""

    def __init__(self, tree):
        self.tree = tree
        self.x = 0

    def _input(self, node_input, value):
        if isinstance(value, (int, float)):
            node_input.default_value = value
        else:
            self.tree.links.new(value, node_input)

    def math(self, op, a, b=0.0, clamp=False, y=0):
        node = self.tree.nodes.new("ShaderNodeMath")
        node.operation = op
        node.use_clamp = clamp
        node.location = (self.x, y)
        self._input(node.inputs[0], a)
        self._input(node.inputs[1], b)
        return node.outputs["Value"]

    def mix(self, data_type, a, b, fac, y=0):
        node = self.tree.nodes.new("ShaderNodeMix")
        node.data_type = data_type
        node.location = (self.x, y)
        kind = "Color" if data_type == "RGBA" else "Float"
        self._input(socket(node.inputs, "Factor_Float"), fac)
        self._input(socket(node.inputs, f"A_{kind}"), a)
        self._input(socket(node.inputs, f"B_{kind}"), b)
        return socket(node.outputs, f"Result_{kind}")


def get_eye_layout_group():
    """
    UV -> per-layer texture coordinates and masks. Everything in eye texture units: U right, D down (= 1 - v).
    """
    group = bpy.data.node_groups.get(EYE_LAYOUT_GROUP)
    if group is None:
        group = bpy.data.node_groups.new(EYE_LAYOUT_GROUP, "ShaderNodeTree")
        for name, default in EYE_DEFAULTS.items():
            sock = group.interface.new_socket(name=name, in_out="INPUT", socket_type="NodeSocketFloat")
            sock.default_value = default
        for layer in SHADER_NAMES:
            group.interface.new_socket(name=f"{layer} uv", in_out="OUTPUT", socket_type="NodeSocketVector")
            group.interface.new_socket(name=f"{layer} mask", in_out="OUTPUT", socket_type="NodeSocketFloat")
    group.nodes.clear()
    b = Builder(group)
    nodes, links = group.nodes, group.links

    inp = nodes.new("NodeGroupInput")
    inp.location = (-1200, 0)
    coords = nodes.new("ShaderNodeTexCoord")
    coords.location = (-1200, -300)
    sep = nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (-1000, -300)
    links.new(coords.outputs["UV"], sep.inputs["Vector"])
    out = nodes.new("NodeGroupOutput")
    out.location = (1200, 0)

    b.x = -800
    u = sep.outputs["X"]
    d = b.math("SUBTRACT", 1.0, sep.outputs["Y"])

    def sprite(layer, cx, cy, w, h, y):
        """texture coords of a sprite centered at (cx, cy), w x h, and a mask for its rectangle"""
        b.x = -600
        w = b.math("MAXIMUM", w, 1e-4, y=y)
        h = b.math("MAXIMUM", h, 1e-4, y=y - 150)
        b.x = -400
        tu = b.math("DIVIDE", b.math("SUBTRACT", u, cx, y=y), w, y=y)
        tt = b.math("DIVIDE", b.math("SUBTRACT", d, cy, y=y - 150), h, y=y - 150)
        b.x = -200
        tu = b.math("ADD", tu, 0.5, y=y)
        tt = b.math("ADD", tt, 0.5, y=y - 150)
        # inside the rectangle: |t - 0.5| <= 0.5 for both
        b.x = 0
        mu = b.math("LESS_THAN", b.math("ABSOLUTE", b.math("SUBTRACT", tu, 0.5, y=y), y=y), 0.5, y=y)
        mt = b.math("LESS_THAN", b.math("ABSOLUTE", b.math("SUBTRACT", tt, 0.5, y=y - 150), y=y - 150), 0.5, y=y - 150)
        b.x = 400
        mask = b.math("MULTIPLY", mu, mt, y=y)
        # image v points up
        tv = b.math("SUBTRACT", 1.0, tt, y=y - 150)
        comb = nodes.new("ShaderNodeCombineXYZ")
        comb.location = (800, y)
        links.new(tu, comb.inputs["X"])
        links.new(tv, comb.inputs["Y"])
        links.new(comb.outputs["Vector"], out.inputs[f"{layer} uv"])
        links.new(mask, out.inputs[f"{layer} mask"])

    b.x = -800
    cx = b.math("ADD", inp.outputs["x"], 0.5, y=400)
    cy = b.math("ADD", inp.outputs["y"], 0.5, y=250)
    sprite("iris", cx, cy, inp.outputs["iris_scale"], inp.outputs["iris_scale"], 400)
    sprite("pupil", cx, cy, inp.outputs["pupil_scale"], inp.outputs["pupil_scale"], 0)
    # lid: full width, top edge at lid - 1
    b.x = -800
    lid_top = b.math("SUBTRACT", inp.outputs["lid"], 1.0, y=-400)
    lid_cy = b.math("ADD", lid_top, b.math("MULTIPLY", inp.outputs["lid_scale"], 0.5, y=-550), y=-550)
    sprite("lid", 0.5, lid_cy, 1.0, inp.outputs["lid_scale"], -400)
    return group


def ensure_eye_props(obj, side):
    for name, default in EYE_DEFAULTS.items():
        prop = f"jak_eye_{side}_{name}"
        if prop not in obj:
            obj[prop] = default
            soft = (-0.5, 0.5) if name in ("x", "y") else (0.0, 1.0)
            obj.id_properties_ui(prop).update(soft_min=soft[0], soft_max=soft[1])


def image_node(tree, image, vector, location, closest=False):
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    node.extension = "EXTEND"
    node.interpolation = "Closest" if closest else "Linear"
    node.location = location
    if isinstance(vector, tuple):
        comb = tree.nodes.new("ShaderNodeCombineXYZ")
        comb.location = (location[0] - 200, location[1])
        for i, v in enumerate(vector):
            comb.inputs[i].default_value = v
        vector = comb.outputs["Vector"]
    tree.links.new(vector, node.inputs["Vector"])
    return node


def fix_eye_material(mat, side, images):
    tree = mat.node_tree
    bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    placeholder = upstream(bsdf.inputs["Base Color"], "TEX_IMAGE") if bsdf else None
    if placeholder is None:
        print(f"  {mat.name}: no placeholder texture found")
        return
    iris_img, pupil_img, lid_img = images
    x0, y0 = placeholder.location.x - 1600, placeholder.location.y

    # parameters from the object's custom properties
    layout = tree.nodes.new("ShaderNodeGroup")
    layout.node_tree = get_eye_layout_group()
    layout.location = (x0 + 300, y0)
    for i, name in enumerate(EYE_DEFAULTS):
        attr = tree.nodes.new("ShaderNodeAttribute")
        attr.attribute_type = "OBJECT"
        attr.attribute_name = f"jak_eye_{side}_{name}"
        attr.location = (x0, y0 - 120 * i)
        tree.links.new(attr.outputs["Fac"], layout.inputs[name])

    w, h = iris_img.size
    bg = image_node(tree, iris_img, (0.5 / w, 1.0 - 0.5 / h, 0.0), (x0 + 700, y0 + 400), closest=True)
    iris = image_node(tree, iris_img, layout.outputs["iris uv"], (x0 + 700, y0 + 100))
    pupil = image_node(tree, pupil_img, layout.outputs["pupil uv"], (x0 + 700, y0 - 200))
    lid = image_node(tree, lid_img, layout.outputs["lid uv"], (x0 + 700, y0 - 500))

    b = Builder(tree)
    b.x = x0 + 1000
    # the eye shader doubles alpha (PS2 range)
    a_bg = b.math("MULTIPLY", bg.outputs["Alpha"], 2.0, clamp=True, y=y0 + 400)
    a_iris = b.math("MULTIPLY", iris.outputs["Alpha"], 2.0, clamp=True, y=y0 + 100)
    a_pupil = b.math("MULTIPLY", pupil.outputs["Alpha"], 2.0, clamp=True, y=y0 - 200)
    a_lid = b.math("MULTIPLY", lid.outputs["Alpha"], 2.0, clamp=True, y=y0 - 500)
    b.x = x0 + 1200
    pupil_fac = b.math("MULTIPLY", a_pupil, layout.outputs["pupil mask"], y=y0 - 200)

    b.x = x0 + 1400
    color = b.mix("RGBA", bg.outputs["Color"], iris.outputs["Color"], layout.outputs["iris mask"], y=y0 + 300)
    alpha = b.mix("FLOAT", a_bg, a_iris, layout.outputs["iris mask"], y=y0 + 100)
    b.x = x0 + 1600
    color = b.mix("RGBA", color, pupil.outputs["Color"], pupil_fac, y=y0 + 300)
    # alpha blend applies to alpha too: a_pupil * a_pupil + a * (1 - a_pupil)
    alpha = b.mix("FLOAT", alpha, a_pupil, pupil_fac, y=y0 + 100)
    b.x = x0 + 1800
    color = b.mix("RGBA", color, lid.outputs["Color"], layout.outputs["lid mask"], y=y0 + 300)
    alpha = b.mix("FLOAT", alpha, a_lid, layout.outputs["lid mask"], y=y0 + 100)

    for link in list(placeholder.outputs["Color"].links):
        tree.links.new(color, link.to_socket)
    for link in list(placeholder.outputs["Alpha"].links):
        tree.links.new(alpha, link.to_socket)

    # envmap mask (jak_merc_materials.py): the strength texture was made from the placeholder's alpha
    sep = next((n for n in tree.nodes if n.type == "SEPARATE_COLOR"), None)
    if sep:
        b.x = x0 + 2000
        mask = b.math("MULTIPLY", alpha, 2.0, clamp=True, y=y0 - 100)
        for link in list(sep.outputs["Blue"].links):
            if link.to_node.type == "BSDF_PRINCIPLED":
                print(f"  {mat.name}: run jak_merc_materials.py first to get the envmap right")
                continue
            tree.links.new(mask, link.to_socket)
    mat["jak_eyes"] = True


def eye_materials(objects):
    """(object, material, side) for eye materials on the objects"""
    for obj in objects:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.node_tree is None:
                continue
            for placeholder, side in PLACEHOLDERS.items():
                if mat.name.startswith(placeholder):
                    yield obj, mat, side


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    glb = argv[0] if argv else None
    if glb:
        bpy.ops.import_scene.gltf(filepath=glb)
        merc_script = Path(__file__).with_name("jak_merc_materials.py")
        sys_argv, sys.argv = sys.argv, [sys.argv[0]]
        exec(compile(merc_script.read_text(), str(merc_script), "exec"), {"__name__": "jak_merc_materials"})
        sys.argv = sys_argv

    game_out = find_game_out_dir(glb)
    if game_out is None:
        print("set GAME_OUT_DIR to your decompiler_out/<game> folder")
        return
    level = Path(glb).parent.name if glb else None

    objects = [o for o in (bpy.context.selected_objects or bpy.data.objects) if o.type == "MESH"]
    images_by_model = {}
    for obj, mat, side in eye_materials(objects):
        ensure_eye_props(obj, side)
        if mat.get("jak_eyes"):
            continue
        # "daxter-highres-lod0", "daxter-highres-lod0_blerc", "daxter-highres-lod0.001" -> "daxter-highres"
        m = re.match(r"(.*?)-lod\d+", obj.name)
        model = m.group(1) if m else obj.name.split(".")[0]
        if model not in images_by_model:
            print(f"{model}:")
            images_by_model[model] = find_eye_images(model, game_out, level)
        images = images_by_model[model]
        if images:
            fix_eye_material(mat, side, images[side])

    if len(argv) > 1:
        bpy.ops.wm.save_as_mainfile(filepath=argv[1])


main()
