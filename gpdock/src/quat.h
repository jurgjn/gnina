// gpdock/quat.h - minimal quaternion / rotation utilities (host + device)
#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define QF __device__
#else
#define QF
#endif

struct qf4 {
  float w, x, y, z;
};

QF inline float qnorm2(float w, float x, float y, float z) {
  return w * w + x * x + y * y + z * z;
}

// rotate point (px,py,pz) by unit quaternion q -> out via (ox,oy,oz)
QF inline void qrot(const float w, const float x, const float y, const float z,
    float px, float py, float pz, float& ox, float& oy, float& oz) {
  // v' = v + 2*q x (q x v + w*v)  (Hamilton)
  // t = 2 * cross(qv, v)  where qv=(x,y,z)
  float tx = 2.0f * (y * pz - z * py);
  float ty = 2.0f * (z * px - x * pz);
  float tz = 2.0f * (x * py - y * px);
  ox = px + w * tx + (y * tz - z * ty);
  oy = py + w * ty + (z * tx - x * tz);
  oz = pz + w * tz + (x * ty - y * tx);
}

// multiply quaternions: r = a * b
QF inline qf4 qmul(const qf4& a, const qf4& b) {
  qf4 r;
  r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
  r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
  r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
  r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
  return r;
}

// small rotation quaternion from axis (ax,ay,az) and angle (radians)
QF inline qf4 qaxis(float ax, float ay, float az, float ang) {
  qf4 r;
  float s = std::sin(ang * 0.5f);
  r.w = std::cos(ang * 0.5f);
  r.x = ax * s;
  r.y = ay * s;
  r.z = az * s;
  return r;
}

QF inline float qnorm(qf4 q) {
  return std::sqrt(qnorm2(q.w, q.x, q.y, q.z));
}
QF inline qf4 qnormd(qf4 q) {
  float n = qnorm(q) + 1e-20f;
  qf4 r;
  r.w = q.w / n;
  r.x = q.x / n;
  r.y = q.y / n;
  r.z = q.z / n;
  return r;
}

// random unit quaternion (given 4 uniform randoms in (0,1))
QF inline qf4 qrand(float u1, float u2, float u3, float u4) {
  qf4 q;
  float a = std::sqrt(1.0f - u1);
  float b = std::sqrt(u1);
  float t1 = 2.0f * 3.14159265358979f * u2;
  float t2 = 2.0f * 3.14159265358979f * u3;
  q.w = a * std::cos(t2);
  q.x = a * std::sin(t2);
  q.y = b * std::sin(t1);
  q.z = b * std::cos(t1);
  return q;
}
