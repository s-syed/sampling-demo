/**
 * Targets.js — Extended target distributions for sampling-demo
 *
 * Each target must implement:
 *   logDensity(x, y)        — log unnormalized density
 *   gradLogDensity(x, y)    — [∂/∂x, ∂/∂y]  (required for MALA/HMC)
 *   xmin, xmax, ymin, ymax  — viewing window bounds (required by Simulation.js)
 */

(function () {
  'use strict';

  // ─── Helpers ────────────────────────────────────────────────────────────────

  function logSumExp(arr) {
    var max = -Infinity;
    for (var i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    if (!isFinite(max)) return -Infinity;
    var sum = 0;
    for (var i = 0; i < arr.length; i++) sum += Math.exp(arr[i] - max);
    return max + Math.log(sum);
  }

  function logMVNdiag(x, y, mux, muy, sx, sy) {
    var dx = x - mux, dy = y - muy;
    return -0.5 * (dx * dx / (sx * sx) + dy * dy / (sy * sy))
           - Math.log(2 * Math.PI * sx * sy);
  }

  function gradLogMVNdiag(x, y, mux, muy, sx, sy) {
    return [-(x - mux) / (sx * sx), -(y - muy) / (sy * sy)];
  }

  // ─── 1. Multimodal Gaussian (equal weights, well-separated) ─────────────────

  MCMC.targets['multimodal-gaussian'] = (function () {
    var K = 4, spread = 3.0, sigma = 0.5;
    var rawW = [1, 1, 1, 1];
    var modes = [], logW = [];
    var wSum = rawW.reduce(function (a, b) { return a + b; }, 0);
    for (var k = 0; k < K; k++) {
      var angle = 2 * Math.PI * k / K;
      modes.push({ x: spread * Math.cos(angle), y: spread * Math.sin(angle) });
      logW.push(Math.log(rawW[k] / wSum));
    }
    return {
      xmin: -6, xmax: 6, ymin: -6, ymax: 6,
      logDensity: function (x, y) {
        var terms = [];
        for (var k = 0; k < K; k++)
          terms.push(logW[k] + logMVNdiag(x, y, modes[k].x, modes[k].y, sigma, sigma));
        return logSumExp(terms);
      },
      gradLogDensity: function (x, y) {
        var logTerms = [];
        for (var k = 0; k < K; k++)
          logTerms.push(logW[k] + logMVNdiag(x, y, modes[k].x, modes[k].y, sigma, sigma));
        var lse = logSumExp(logTerms);
        var gx = 0, gy = 0;
        for (var k = 0; k < K; k++) {
          var r = Math.exp(logTerms[k] - lse);
          var g = gradLogMVNdiag(x, y, modes[k].x, modes[k].y, sigma, sigma);
          gx += r * g[0]; gy += r * g[1];
        }
        return [gx, gy];
      }
    };
  }());

  // ─── 2. Multimodal Gaussian (unequal weights) ───────────────────────────────

  MCMC.targets['multimodal-unequal'] = (function () {
    var components = [
      { x: -4.0, y:  0.0, sx: 0.6, sy: 0.6, w: 0.5  },
      { x:  4.0, y:  0.0, sx: 0.4, sy: 0.4, w: 0.3  },
      { x:  0.0, y:  4.0, sx: 0.5, sy: 0.5, w: 0.15 },
      { x:  0.0, y: -4.0, sx: 0.3, sy: 0.3, w: 0.05 },
    ];
    var wSum = components.reduce(function (a, c) { return a + c.w; }, 0);
    var logW = components.map(function (c) { return Math.log(c.w / wSum); });
    return {
      xmin: -6, xmax: 6, ymin: -6, ymax: 6,
      logDensity: function (x, y) {
        var terms = [];
        for (var k = 0; k < components.length; k++) {
          var c = components[k];
          terms.push(logW[k] + logMVNdiag(x, y, c.x, c.y, c.sx, c.sy));
        }
        return logSumExp(terms);
      },
      gradLogDensity: function (x, y) {
        var logTerms = [];
        for (var k = 0; k < components.length; k++) {
          var c = components[k];
          logTerms.push(logW[k] + logMVNdiag(x, y, c.x, c.y, c.sx, c.sy));
        }
        var lse = logSumExp(logTerms);
        var gx = 0, gy = 0;
        for (var k = 0; k < components.length; k++) {
          var r = Math.exp(logTerms[k] - lse);
          var c = components[k];
          var g = gradLogMVNdiag(x, y, c.x, c.y, c.sx, c.sy);
          gx += r * g[0]; gy += r * g[1];
        }
        return [gx, gy];
      }
    };
  }());

  // ─── 3. GMM-32 (Blessing et al. 2208.01893) ─────────────────────────────────

  MCMC.targets['gmm32'] = (function () {
    var modes = [];
    var rows = 4, cols = 8, spacing = 3.0, sigma = 0.3;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        modes.push({
          x: (c - (cols - 1) / 2) * spacing,
          y: (r - (rows - 1) / 2) * spacing
        });
      }
    }
    var logWk = Math.log(1 / modes.length);
    return {
      xmin: -14, xmax: 14, ymin: -6, ymax: 6,
      logDensity: function (x, y) {
        var terms = [];
        for (var k = 0; k < modes.length; k++)
          terms.push(logWk + logMVNdiag(x, y, modes[k].x, modes[k].y, sigma, sigma));
        return logSumExp(terms);
      },
      gradLogDensity: function (x, y) {
        var logTerms = [];
        for (var k = 0; k < modes.length; k++)
          logTerms.push(logWk + logMVNdiag(x, y, modes[k].x, modes[k].y, sigma, sigma));
        var lse = logSumExp(logTerms);
        var gx = 0, gy = 0;
        for (var k = 0; k < modes.length; k++) {
          var r = Math.exp(logTerms[k] - lse);
          var g = gradLogMVNdiag(x, y, modes[k].x, modes[k].y, sigma, sigma);
          gx += r * g[0]; gy += r * g[1];
        }
        return [gx, gy];
      }
    };
  }());

  // ─── 4. Ring / Donut ────────────────────────────────────────────────────────

  MCMC.targets['ring'] = (function () {
    var R = 3.0, sigma = 0.5;
    return {
      xmin: -6, xmax: 6, ymin: -6, ymax: 6,
      logDensity: function (x, y) {
        var r = Math.sqrt(x * x + y * y);
        var d = r - R;
        return -0.5 * d * d / (sigma * sigma);
      },
      gradLogDensity: function (x, y) {
        var r = Math.sqrt(x * x + y * y);
        if (r < 1e-10) return [0, 0];
        var d = r - R;
        var scale = -d / (sigma * sigma * r);
        return [scale * x, scale * y];
      }
    };
  }());

  // ─── 5. Twisted Banana ──────────────────────────────────────────────────────

  MCMC.targets['funnel'] = (function () {
    var b = 0.05, sigma1 = 2.0, sigma2 = 1.0;
    return {
      xmin: -6, xmax: 6, ymin: -4, ymax: 4,
      logDensity: function (x, y) {
        var shift = b * x * x;
        return -0.5 * x * x / (sigma1 * sigma1)
               -0.5 * (y - shift) * (y - shift) / (sigma2 * sigma2);
      },
      gradLogDensity: function (x, y) {
        var shift = b * x * x;
        var dy = (y - shift) / (sigma2 * sigma2);
        var gx = -(x / (sigma1 * sigma1) + dy * 2 * b * x);
        var gy = dy;
        return [gx, gy];
      }
    };
  }());

  // ─── 6. Rosenbrock ──────────────────────────────────────────────────────────

  MCMC.targets['rosenbrock'] = (function () {
    var scale = 20.0;
    return {
      xmin: -3, xmax: 3, ymin: -2, ymax: 8,
      logDensity: function (x, y) {
        var a = 1 - x;
        var b = y - x * x;
        return -(a * a + 100 * b * b) / scale;
      },
      gradLogDensity: function (x, y) {
        var b = y - x * x;
        var gx = (2 * (1 - x) + 400 * x * b) / scale;
        var gy = -200 * b / scale;
        return [gx, gy];
      }
    };
  }());

  // ─── 7. Bivariate Student-t ─────────────────────────────────────────────────

  MCMC.targets['student-t'] = (function () {
    var nu = 2.0;
    return {
      xmin: -8, xmax: 8, ymin: -8, ymax: 8,
      logDensity: function (x, y) {
        var r2 = x * x + y * y;
        return -0.5 * (nu + 2) * Math.log(1 + r2 / nu);
      },
      gradLogDensity: function (x, y) {
        var r2 = x * x + y * y;
        var scale = -(nu + 2) / (nu + r2);
        return [scale * x, scale * y];
      }
    };
  }());

  // ─── 8. Neal's Funnel ───────────────────────────────────────────────────────

  MCMC.targets['neals-funnel'] = (function () {
    return {
      xmin: -8, xmax: 8, ymin: -6, ymax: 6,
      logDensity: function (x, y) {
        var logp_v = -0.5 * y * y / 9;
        var var_x  = Math.exp(y);
        var logp_x = -0.5 * x * x / var_x - 0.5 * y;
        return logp_v + logp_x;
      },
      gradLogDensity: function (x, y) {
        var evy = Math.exp(y);
        var gx = -x / evy;
        var gy = -y / 9 + 0.5 * x * x / evy - 0.5;
        return [gx, gy];
      }
    };
  }());

  // ─── 9. Double Banana ───────────────────────────────────────────────────────

  MCMC.targets['double-banana'] = (function () {
    var b = 0.1, sigma = 0.5;
    function logComp(x, y, flip) {
      var y0 = flip * b * x * x;
      var dy = y - y0;
      return -0.5 * x * x - 0.5 * dy * dy / (sigma * sigma);
    }
    function gradLogComp(x, y, flip) {
      var y0 = flip * b * x * x;
      var dy = y - y0;
      var gx = -x - dy * flip * 2 * b * x / (sigma * sigma);
      var gy = dy / (sigma * sigma);
      return [gx, gy];
    }
    return {
      xmin: -5, xmax: 5, ymin: -4, ymax: 4,
      logDensity: function (x, y) {
        return logSumExp([logComp(x, y, 1) + Math.log(0.5),
                          logComp(x, y, -1) + Math.log(0.5)]);
      },
      gradLogDensity: function (x, y) {
        var t1 = logComp(x, y,  1) + Math.log(0.5);
        var t2 = logComp(x, y, -1) + Math.log(0.5);
        var lse = logSumExp([t1, t2]);
        var r1 = Math.exp(t1 - lse), r2 = Math.exp(t2 - lse);
        var g1 = gradLogComp(x, y,  1);
        var g2 = gradLogComp(x, y, -1);
        return [r1 * g1[0] + r2 * g2[0], r1 * g1[1] + r2 * g2[1]];
      }
    };
  }());

}());
