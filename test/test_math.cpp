#include "common/math/geometry.h"

#include "gtest/gtest.h"

TEST(Math, RaySphereOrigin) {
  math::Vector<float, 3> sphere(0, 0, 0);
  math::Vector<float, 3> line_origin(0, 0, 0);
  math::Vector<float, 3> line_dir(1, 2, 3);  // shouldn't matter
  auto result = math::ray_sphere_intersect(line_origin, line_dir, sphere, 1.f);
  EXPECT_FLOAT_EQ(result.u[0], 1.f);
  EXPECT_FLOAT_EQ(result.u[1], -1.f);
}

TEST(Math, RaySphereOriginOffsetXProbeX) {
  math::Vector<float, 3> sphere(0, 0, 0);
  math::Vector<float, 3> line_origin(0.5, 0, 0);
  math::Vector<float, 3> line_dir(1, 0, 0);
  auto result = math::ray_sphere_intersect(line_origin, line_dir, sphere, 1.f);
  EXPECT_FLOAT_EQ(result.u[0], 0.5f);
  EXPECT_FLOAT_EQ(result.u[1], -1.5f);
}

TEST(Math, RaySphereOriginOffsetXProbeY) {
  math::Vector<float, 3> sphere(0, 0, 0);
  math::Vector<float, 3> line_origin(0.5, 0, 0);
  math::Vector<float, 3> line_dir(0, 1, 0);
  auto result = math::ray_sphere_intersect(line_origin, line_dir, sphere, 1.f);
  // should be +/- sqrt(3)/2, draw a triangle.
  EXPECT_FLOAT_EQ(result.u[0], std::sqrt(3.f) / 2.f);
  EXPECT_FLOAT_EQ(result.u[1], -std::sqrt(3.f) / 2.f);
}

TEST(Math, RaySphereMakeTestCaseHit) {
  math::Vector<float, 3> sphere(0.4, 0.3, 0.7);
  math::Vector<float, 3> line_origin(0.2, -3, 0);
  math::Vector<float, 3> line_dir(0, 1, 0);
  auto result = math::ray_sphere_intersect(line_origin, line_dir, sphere, 1.2f);
  fmt::print("Test case: {} {} {}\n", result.hit, result.u[0], result.u[1]);
}

TEST(Math, RaySphereMiss) {
  math::Vector<float, 3> sphere(1, 2, 3);
  fmt::print("{}\n", sphere.to_string_aligned());
}

TEST(Math, TransformRotation) {
  math::Transform t = math::Transform::identity();
  math::Quaternionf q(0, 0.7071068, 0, 0.7071068);
  t.set_rotation(q);
  EXPECT_TRUE(t.rotation().approx_equal(q));
}

TEST(Math, TransformScale) {
  math::Transform t = math::Transform::identity();
  math::Vector3f s(2, 3, 4);
  t.set_scale(s);
  EXPECT_TRUE(t.scale().approx_equal(s));
  math::Transform t2 = t;
  t2.set_scale(t2.scale());
  EXPECT_TRUE(t2.approx_equal(t));
}

TEST(Math, TransformScaleThenRotationAndFlipScale) {
  math::Transform t = math::Transform::identity();
  math::Quaternionf q(0, 0.7071068, 0, 0.7071068);
  math::Vector3f s(2, 3, 4);
  t.set_scale(s);
  t.set_rotation(q);
  EXPECT_TRUE(t.rotation().approx_equal(q));
  EXPECT_TRUE(t.scale().approx_equal(s));
  s.x() = -6;
  t.set_scale(s);
  t.set_rotation(q);
  EXPECT_TRUE(t.rotation().approx_equal(q));
  EXPECT_TRUE(t.scale().approx_equal(s));
}

TEST(Math, TransformTranslation) {
  math::Transform t = math::Transform::identity();
  t.set_translation(math::Vector3f(2, 3, 4));
  EXPECT_TRUE(t.translation().approx_equal(math::Vector3f(2, 3, 4)));
}

TEST(Math, TransformDirectionCheck) {
  math::Transform t = math::Transform::identity();
  math::Quaternionf q(0, 0.7071068, 0, 0.7071068);
  t.set_rotation(q);
  math::Vector4f dir(1, 0, 0, 0);
  auto transformed_dir = t * dir;
  EXPECT_TRUE(transformed_dir.approx_equal(math::Vector4f(0, 0, -1, 0)));
}

TEST(Math, TRS) {
  math::Vector3f trans(1, 2, 3);
  math::Quaternionf rot(0, 0.7071068, 0, 0.7071068);
  math::Vector3f scale(2, 2, 2);
  auto m = math::Transform::from_trs(trans, rot, scale);
  fmt::print("Matrix: \n{}\n", m.to_string_aligned());
}

TEST(Math, TRSRoundTrip) {
  math::Vector3f t(1, 2, 3);
  math::Vector3f s(-6, 4, 1);
  auto check = [&](const char* name, const math::Quaternionf& q) {
    SCOPED_TRACE(name);
    auto m = math::Transform::from_trs(t, q, s);
    EXPECT_TRUE(m.translation().approx_equal(t));
    EXPECT_TRUE(m.rotation().approx_equal(q));
    EXPECT_TRUE(m.scale().approx_equal(s));
    EXPECT_TRUE(
        math::Transform::from_trs(m.translation(), m.rotation(), m.scale()).approx_equal(m));
  };
  check("x", math::Quaternionf(1, 0, 0, 0));
  check("y", math::Quaternionf(0, 1, 0, 0));
  check("z", math::Quaternionf(0, 0, 1, 0));
  check("y90", math::Quaternionf(0, 0.7071068, 0, 0.7071068));
  check("nonalignedRot", math::Quaternionf(0.1, 0.2, 0.3, 0.4).normalized());
}
