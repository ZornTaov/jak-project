"""
Fix up materials of models exported by the decompiler's glTF exporter (fr3_to_gltf.cpp) so they look
like the game in Blender (4.2+).

The game draws envmapped merc models in two passes:
  1. diffuse:  color = texture * vertex_color * 2         (PS2: 0x80 means 1.0)
  2. envmap:   color += envmap * fade * 2 * dest_alpha    (dest_alpha = diffuse texture alpha * 2)
The envmap is sampled with view-dependent sphere-map UVs (see emerc.vert), not mesh UVs.

"fade" is computed per effect, every frame, in foreground.gc:
  fade = clamp(fade0 * dist + fade1, 0, 1) / 128 * tint * current-env-color
where tint is per-effect art data (mei-envmap-tint, not exported) and current-env-color comes from the level's
mood lights (time-of-day.gc, update-environment-colors):
  env = 48 * (sun + 0.5 * (ambient + sun))
That's what tints the envmap with the scene light color (e.g. the sunrise in the intro). Here it's the shared
"JakEnvColor" node group: edit its Sun / Ambient colors once and every envmapped material follows.
Effects with the ignore-tod-for-envmap-tint bit skip the env color in the game; that bit isn't exported either.

The exporter encodes this as:
  - baseColorTexture             = diffuse texture (alpha is PS2 range, 0x80 = 1.0)
  - COLOR_0                      = vertex color (PS2 range, 0x80 = 1.0), sRGB-encoded so Blender's byte color
                                   attribute holds the PS2 byte exactly; alpha is linear (PS2 range too)
  - metallicRoughnessTexture     = "<tex>-envmap-strength", blue = min(1, alpha * 2), green = 0
  - KHR_materials_specular color = the envmap texture
which Blender imports as a mirror-smooth metal with the envmap as Specular Tint. This script rewires that:
  - vertex color is scaled by 2 ^ 2.2 (the game's x2 happens in gamma space)
  - diffuse alpha and vertex alpha are each scaled by 2 (game: T0.a * vtx.a * 2 * 2)
  - envmap materials: metallic/roughness/specular tint unlinked, envmap sampled with sphere-map UVs,
    multiplied by the mask and added through Emission.

Usage:
  - In Blender: import the .glb, select the objects (or nothing, for all materials), run this script.
  - Command line: blender -b --python jak_merc_materials.py -- model.glb [out.blend]
  - Exporting back to glb: Data > Mesh > Vertex Color must be "Active" (or "Name"), not the default "Material".
    The nodes added here hide the color attribute from the exporter's "used by material" check, and it then writes
    a fake all-white COLOR_0. Also enable Include > Custom Properties so the og_* material extras are kept.
"""

import sys

import bpy

# "game": every envmap is an additive matcap, like the game.
# "pbr": envmaps that use a generic environment texture (goggles, metal) become Principled metallic instead,
#        so they react to Blender lights / HDRIs. Character-specific "hilite" envmaps stay additive.
MODE = "game"

# Default per-effect envmap tint (mei-envmap-tint), as rgb bytes. Not exported, so 0x80 for everything;
# tweak per material on the "Effect Tint" node afterwards.
EFFECT_TINT = (0x80, 0x80, 0x80)

# Default scene light colors for the JakEnvColor group (game mood light colors, 1.0 = full).
# sun = ambient = 1.0 gives env = 96, the game's default (mood.gc).
SUN_COLOR = (1.0, 1.0, 1.0)
AMBIENT_COLOR = (1.0, 1.0, 1.0)

# If set to the name of a light object, JakEnvColor's sun color is driven by that light's color.
SUN_LIGHT = None

# Sphere map orientation. On the PS2 both camera-space Y and texture T point down; in Blender both camera-space Y
# and image V point up, so no flip should be needed. Flip if the highlights look mirrored compared to the game.
FLIP_U = False
FLIP_V = False

# Undo the PS2 half-range vertex colors and alpha (0x80 = 1.0).
FIX_VERTEX_COLOR = True

# The game's x2 applies in gamma space; on the linear attribute value that's ~2 ^ 2.2.
VERTEX_COLOR_SCALE = 2.0**2.2

# Envmap textures with these prefixes are treated as "metal" in "pbr" mode.
METAL_ENVMAP_PREFIXES = ("pal-environment", "environment-")

SPHERE_MAP_GROUP = "JakSphereMap"
ENV_COLOR_GROUP = "JakEnvColor"


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


def unlink(tree, sock):
    for link in list(sock.links):
        tree.links.remove(link)


def get_sphere_map_group():
    """Node group computing sphere map UVs from the view space reflection vector (emerc.vert)."""
    group = bpy.data.node_groups.get(SPHERE_MAP_GROUP)
    if group is None:
        group = bpy.data.node_groups.new(SPHERE_MAP_GROUP, "ShaderNodeTree")
        group.interface.new_socket(name="Vector", in_out="OUTPUT", socket_type="NodeSocketVector")
    # rebuild every time, so changes to FLIP_U/FLIP_V apply
    group.nodes.clear()
    nodes, links = group.nodes, group.links

    coords = nodes.new("ShaderNodeTexCoord")
    coords.location = (-1000, 0)

    to_camera = nodes.new("ShaderNodeVectorTransform")
    to_camera.vector_type = "VECTOR"
    to_camera.convert_from = "WORLD"
    to_camera.convert_to = "CAMERA"
    to_camera.location = (-800, 0)
    links.new(coords.outputs["Reflection"], to_camera.inputs["Vector"])

    # sphere map: m = normalize(r + toward_viewer). Blender's camera looks down +Z in camera space.
    add_view = nodes.new("ShaderNodeVectorMath")
    add_view.operation = "ADD"
    add_view.inputs[1].default_value = (0.0, 0.0, -1.0)
    add_view.location = (-600, 0)
    links.new(to_camera.outputs["Vector"], add_view.inputs[0])

    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    normalize.location = (-400, 0)
    links.new(add_view.outputs["Vector"], normalize.inputs[0])

    # uv = m.xy * 0.5 + 0.5
    to_uv = nodes.new("ShaderNodeVectorMath")
    to_uv.operation = "MULTIPLY_ADD"
    to_uv.inputs[1].default_value = (0.5, 0.5, 0.0)
    to_uv.inputs[2].default_value = (0.5, 0.5, 0.0)
    to_uv.location = (-200, 0)
    links.new(normalize.outputs["Vector"], to_uv.inputs[0])

    uv = to_uv.outputs["Vector"]
    if FLIP_U or FLIP_V:
        # uv * (-1 or 1) + (1 or 0)
        flip = nodes.new("ShaderNodeVectorMath")
        flip.operation = "MULTIPLY_ADD"
        flip.inputs[1].default_value = (-1.0 if FLIP_U else 1.0, -1.0 if FLIP_V else 1.0, 0.0)
        flip.inputs[2].default_value = (1.0 if FLIP_U else 0.0, 1.0 if FLIP_V else 0.0, 0.0)
        flip.location = (0, 0)
        links.new(uv, flip.inputs[0])
        uv = flip.outputs["Vector"]

    out = nodes.new("NodeGroupOutput")
    out.location = (200, 0)
    links.new(uv, out.inputs["Vector"])
    return group


def get_env_color_group():
    """
    Node group with the scene's envmap color, as an envmap multiplier: env / 128, so the game's default
    env of 96 gives 0.75. Created once and then left alone, so edits to its Sun / Ambient colors survive re-runs.
    """
    group = bpy.data.node_groups.get(ENV_COLOR_GROUP)
    if group is not None:
        return group
    group = bpy.data.node_groups.new(ENV_COLOR_GROUP, "ShaderNodeTree")
    group.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    nodes, links = group.nodes, group.links

    sun = nodes.new("ShaderNodeRGB")
    sun.name = sun.label = "Sun Color"
    sun.outputs["Color"].default_value = (*SUN_COLOR, 1.0)
    sun.location = (-600, 100)

    ambient = nodes.new("ShaderNodeRGB")
    ambient.name = ambient.label = "Ambient Color"
    ambient.outputs["Color"].default_value = (*AMBIENT_COLOR, 1.0)
    ambient.location = (-600, -150)

    # 48 * (sun + 0.5 * (ambient + sun)) / 128 = 0.5625 * sun + 0.1875 * ambient
    sun_part = nodes.new("ShaderNodeVectorMath")
    sun_part.operation = "SCALE"
    sun_part.inputs["Scale"].default_value = 0.5625
    sun_part.location = (-350, 100)
    links.new(sun.outputs["Color"], sun_part.inputs[0])

    env = nodes.new("ShaderNodeVectorMath")
    env.operation = "MULTIPLY_ADD"
    env.inputs[1].default_value = (0.1875, 0.1875, 0.1875)
    env.location = (-150, 0)
    links.new(ambient.outputs["Color"], env.inputs[0])
    links.new(sun_part.outputs["Vector"], env.inputs[2])

    out = nodes.new("NodeGroupOutput")
    out.location = (100, 0)
    links.new(env.outputs["Vector"], out.inputs["Color"])

    if SUN_LIGHT:
        light = bpy.data.objects.get(SUN_LIGHT)
        if light is None or light.type != "LIGHT":
            print(f"SUN_LIGHT {SUN_LIGHT!r} is not a light object, not adding drivers")
        else:
            for i in range(3):
                fcurve = sun.outputs["Color"].driver_add("default_value", i)
                var = fcurve.driver.variables.new()
                var.name = "c"
                var.targets[0].id_type = "LIGHT"
                var.targets[0].id = light.data
                var.targets[0].data_path = f"color[{i}]"
                fcurve.driver.expression = "c"
    return group


def fix_vertex_color(tree, attr):
    """
    vertex color: the attribute holds the PS2 byte v as sRGB, so it reads as linear ~(v / 255) ^ 2.2.
    The game multiplies by 2 * v / 255 in gamma space, i.e. a linear factor of ~(2 * v / 255) ^ 2.2 = attr * 2 ^ 2.2.
    """
    for link in list(attr.outputs["Color"].links):
        target = link.to_socket
        tree.links.remove(link)

        scale = tree.nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.label = "PS2 x2 (gamma space)"
        scale.inputs["Scale"].default_value = VERTEX_COLOR_SCALE
        scale.location = (attr.location.x + 200, attr.location.y - 150)
        tree.links.new(attr.outputs["Color"], scale.inputs[0])
        tree.links.new(scale.outputs["Vector"], target)


def fix_vertex_alpha(tree, attr):
    """vertex alpha: PS2 range (0x80 = 1.0), linear -> x2. Not clamped: the game clamps the final product."""
    targets = [link.to_socket for link in attr.outputs["Alpha"].links]
    if not targets:
        return
    double = tree.nodes.new("ShaderNodeMath")
    double.operation = "MULTIPLY"
    double.inputs[1].default_value = 2.0
    double.location = (attr.location.x + 200, attr.location.y - 350)
    unlink(tree, attr.outputs["Alpha"])
    tree.links.new(attr.outputs["Alpha"], double.inputs[0])
    for target in targets:
        tree.links.new(double.outputs["Value"], target)


def fix_alpha(tree, base_tex):
    """diffuse alpha: PS2 range -> x2 before the importer's alpha clip"""
    alpha = base_tex.outputs["Alpha"]
    targets = [link.to_socket for link in alpha.links]
    if not targets:
        return
    double = tree.nodes.new("ShaderNodeMath")
    double.operation = "MULTIPLY"
    double.use_clamp = True
    double.inputs[1].default_value = 2.0
    double.location = (base_tex.location.x + 300, base_tex.location.y - 250)
    unlink(tree, alpha)
    tree.links.new(alpha, double.inputs[0])
    for target in targets:
        tree.links.new(double.outputs["Value"], target)


def fix_envmap(tree, bsdf, env_tex):
    sep = upstream(bsdf.inputs["Metallic"], "SEPARATE_COLOR")
    if sep is None:
        print("  no envmap strength texture, skipping envmap")
        return
    mask = sep.outputs["Blue"]  # min(1, diffuse alpha * 2)

    for name in ("Metallic", "Roughness", "Specular Tint"):
        unlink(tree, bsdf.inputs[name])
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 1.0  # same as the non-envmapped materials
    bsdf.inputs["Specular Tint"].default_value = (1.0, 1.0, 1.0, 1.0)

    # envmap sampled with sphere map UVs
    sphere = tree.nodes.new("ShaderNodeGroup")
    sphere.node_tree = get_sphere_map_group()
    sphere.location = (env_tex.location.x - 250, env_tex.location.y)
    unlink(tree, env_tex.inputs["Vector"])
    tree.links.new(sphere.outputs["Vector"], env_tex.inputs["Vector"])
    env_tex.extension = "EXTEND"

    # envmap * mask * env color * effect tint -> emission (additive, like SRC_0_DST_DST)
    def multiply(a, x, label=""):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.label = label
        socket(mix.inputs, "Factor_Float").default_value = 1.0
        mix.location = (env_tex.location.x + x, env_tex.location.y)
        tree.links.new(a, socket(mix.inputs, "A_Color"))
        return mix

    masked = multiply(env_tex.outputs["Color"], 300)
    tree.links.new(mask, socket(masked.inputs, "B_Color"))

    env_color = tree.nodes.new("ShaderNodeGroup")
    env_color.node_tree = get_env_color_group()
    env_color.location = (env_tex.location.x + 300, env_tex.location.y - 250)
    lit = multiply(socket(masked.outputs, "Result_Color"), 500)
    tree.links.new(env_color.outputs["Color"], socket(lit.inputs, "B_Color"))

    # fade = tint * env / 128, shader does * 2 / 255
    tinted = multiply(socket(lit.outputs, "Result_Color"), 700, "Effect Tint")
    socket(tinted.inputs, "B_Color").default_value = (*(c * 2 / 255 for c in EFFECT_TINT), 1.0)

    tree.links.new(socket(tinted.outputs, "Result_Color"), bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 1.0

    env_name = env_tex.image.name if env_tex.image else ""
    if MODE == "pbr" and env_name.startswith(METAL_ENVMAP_PREFIXES):
        tree.links.new(mask, bsdf.inputs["Metallic"])
        bsdf.inputs["Roughness"].default_value = 0.25
        bsdf.inputs["Emission Strength"].default_value = 0.5


def fix_material(mat):
    # the marker lives on the node tree, not the material: material custom properties get exported as glTF extras
    # (needed for og_*), and would come back on re-import and skip the fix on a fresh, unfixed node tree.
    if mat.node_tree is None or mat.node_tree.get("jak_fixed"):
        return
    tree = mat.node_tree
    bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    base_tex = upstream(bsdf.inputs["Base Color"], "TEX_IMAGE")
    env_tex = upstream(bsdf.inputs["Specular Tint"], "TEX_IMAGE")
    print(f"{mat.name}: envmap={env_tex.image.name if env_tex and env_tex.image else None}")

    attr = upstream(bsdf.inputs["Base Color"], "VERTEX_COLOR")
    if FIX_VERTEX_COLOR and attr:
        fix_vertex_color(tree, attr)
    if env_tex:
        # alpha is the envmap mask here, not transparency
        fix_envmap(tree, bsdf, env_tex)
    else:
        if base_tex:
            fix_alpha(tree, base_tex)
        if FIX_VERTEX_COLOR and attr:
            fix_vertex_alpha(tree, attr)
    tree["jak_fixed"] = True


def selected_materials():
    objects = bpy.context.selected_objects
    if not objects:
        return list(bpy.data.materials)
    mats = []
    for obj in objects:
        for slot in obj.material_slots:
            if slot.material and slot.material not in mats:
                mats.append(slot.material)
    return mats


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if argv:
        bpy.ops.import_scene.gltf(filepath=argv[0])
    for mat in selected_materials():
        fix_material(mat)
    if len(argv) > 1:
        bpy.ops.wm.save_as_mainfile(filepath=argv[1])


main()
