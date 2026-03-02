/**
 * Targets.js — Extended target distributions for sampling-demo
 *
 * Drop-in extension for chi-feng/mcmc-demo. Each target implements:
 *   logDensity(x, y)           — returns log unnormalized density (number)
 *   gradLogDensity(x, y)       — returns [∂/∂x, ∂/∂y] (required for MALA/HMC)
 *
 * To add a new target later:
 *   1. Define it below following the same pattern.
 *   2. Register it at the bottom with MCMC.targets['your-key'] = { ... };
 *   3. It will appear automatically in the dropdown.
 *
 * Author: added for s-syed/sampling-demo
 */

(function () {
  'use strict';

  // ─── Helpers ────────────────────────────────────────────────────────────────

  /** log of sum of exponentials (numerically stable) */
  function logSumExp(arr) {
    var max = -Infinity;
    for (var i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    if (!isFinite(max)) return -Infinity;
    var sum = 0;
    for (var i = 0; i < arr.length; i++) sum += Math.exp(arr[i] - max);
    return max + Math.log(sum);
  }

  /** Multivariate normal log-density (2D, diagonal covariance) */
  function logMVNdiag(x, y, mux, muy, sx, sy) {
    var dx = x - mux, dy = y - muy;
    return -0.5 * (dx * dx / (sx * sx) + dy * dy / (sy * sy))
           - Math.log(2 * Math.PI * sx * sy);
  }

  /** Grad of logMVNdiag w.r.t. (x, y) */
  function gradLogMVNdiag(x, y, mux, muy, sx, sy) {
    return [-(x - mux) / (sx * sx), -(y - muy) / (sy * sy)];
  }

  // ─── 1. Well-Separated Multimodal Gaussian (configurable weights) ────────────
  //
  //   A mixture of K Gaussians placed on a circle of radius `spread`.
  //   Weights can be non-uniform to illustrate asymmetric mixing difficulty.
  //   Default: 4 modes, equal weights, spread = 3.
  //
  //   Useful for: showing how RWM gets trapped, HMC can tunnel, PT effect.

  MCMC.targets['multimodal-gaussian'] = (function () {
    // --- Parameters (edit these to change the visualisation) ---
    var K      = 4;          // number of modes
    var spread = 3.0;        // radius of mode circle
    var sigma  = 0.5;        // std of each component
    // Non-uniform weights (will be normalised); try [4,1,1,1] to break symmetry
    var rawW   = [1, 1, 1, 1];

    // --- Derived quantities (do not edit) ---
    var modes = [], logW = [];
    var wSum = rawW.reduce(function (a, b) { return a + b; }, 0);
    for (var k = 0; k < K; k++) {
      var angle = 2 * Math.PI * k / K;
      modes.push({ x: spread * Math.cos(angle), y: spread * Math.sin(angle) });
      logW.push(Math.log(rawW[k] / wSum));
    }

    return {
      logDensity: function (x, y) {
        var terms = [];
        for (var k = 0; k < K; k++)
          terms.push(logW[k] + logMVNdiag(x, y, modes[k].x, modes[k].y, sigma, sigma));
        return logSumExp(terms);
      },
      gradLogDensity: function (x, y) {
        // ∇ log p = Σ_k r_k ∇ log φ_k   where r_k = softmax responsibilities
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

  // ─── 2. Unequal-Weight Multimodal (asymmetric, well-separated) ──────────────
  //
  //   Same structure as above but with strongly unequal weights and larger
  //   separation, making this a harder mixing problem.

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

  // ─── 3. GMM-32 — hard benchmark from Blessing et al. 2208.01893 ─────────────
  //
  //   A 2D mixture of 32 Gaussians with small variance arranged in a 4×8 grid,
  //   all with equal weight. Used as a hard benchmark for annealing methods.
  //   Modes are well-separated (spacing ~2.5 σ_between ≫ σ_within).
  //
  //   Reference: Blessing et al. "Beyond ELBOs: A Large-Scale Evaluation of
  //   Variational Methods for Sampling" (2023) arXiv:2208.01893

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
  //
  //   p(x,y) ∝ exp( -(r - R)² / (2σ²) )  where r = sqrt(x²+y²)
  //
  //   A challenging target for gradient-based methods: the gradient points
  //   radially, so MALA/HMC must orbit the ring; illustrates the value of
  //   long trajectories.

  MCMC.targets['ring'] = (function () {
    var R = 3.0;   // ring radius
    var sigma = 0.5; // ring width

    return {
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

  // ─── 5. Banana / Neal's Funnel (2D twisted Gaussian) ─────────────────────────
  //
  //   The classic "banana-shaped" distribution:
  //     x₁ ~ N(0,1),   x₂ | x₁ ~ N(bx₁², 1)
  //   Controlled by curvature b (default 0.03 matches Neal 2003).
  //
  //   Different from the "banana" already in the original repo (which uses the
  //   Rosenbrock parameterisation). This version makes the vertical spread more
  //   pronounced.

  MCMC.targets['funnel'] = (function () {
    var b = 0.05;   // curvature; try 0.1 for a sharper bend
    var sigma1 = 2.0, sigma2 = 1.0;

    return {
      logDensity: function (x, y) {
        // x plays the role of x₁ (the "wide" axis)
        var shift = b * x * x;
        return -0.5 * x * x / (sigma1 * sigma1)
               -0.5 * (y - shift) * (y - shift) / (sigma2 * sigma2);
      },
      gradLogDensity: function (x, y) {
        var shift = b * x * x;
        var dy = (y - shift) / (sigma2 * sigma2);
        var dx = x / (sigma1 * sigma1) + dy * 2 * b * x;  // chain rule on shift
        return [-dx, -(-dy)]; // note: returning [∂logp/∂x, ∂logp/∂y]
      }
    };
  }());

  // ─── 6. Rosenbrock ──────────────────────────────────────────────────────────
  //
  //   p(x,y) ∝ exp( -( (1-x)² + 100(y-x²)² ) / scale )
  //
  //   The classic Rosenbrock "banana valley". The probability mass follows
  //   a narrow curved valley, making this a canonical hard target for MCMC.
  //   `scale` controls how peaked the distribution is.

  MCMC.targets['rosenbrock'] = (function () {
    var scale = 20.0;  // lower = narrower valley = harder

    return {
      logDensity: function (x, y) {
        var a = 1 - x;
        var b = y - x * x;
        return -(a * a + 100 * b * b) / scale;
      },
      gradLogDensity: function (x, y) {
        var b = y - x * x;
        var dlogp_dx = (2 * (1 - x) + 100 * 2 * b * 2 * x) / scale; // -(d/dx of numer)/scale with sign flip
        var dlogp_dy = -(200 * b * (-1)) / scale;                     // careful signs
        // Let's redo carefully:
        // logp = -(  (1-x)^2 + 100*(y-x^2)^2  ) / scale
        // ∂logp/∂x = -[ 2(1-x)(-1) + 100*2(y-x^2)(-2x) ] / scale
        //           = [ 2(1-x) + 400x(y-x^2) ] / scale
        // ∂logp/∂y = -[ 100*2(y-x^2) ] / scale = -200(y-x^2)/scale
        var gx = (2 * (1 - x) + 400 * x * b) / scale;
        var gy = -200 * b / scale;
        return [gx, gy];
      }
    };
  }());

  // ─── 7. Heavy-Tailed (Bivariate Student-t) ──────────────────────────────────
  //
  //   p(x,y) ∝ (1 + (x²+y²)/ν)^{-(ν+2)/2}
  //
  //   With ν small (default 2), the tails are heavy. Good for illustrating
  //   that RWM/HMC tuned for Gaussian assumptions can misbehave.

  MCMC.targets['student-t'] = (function () {
    var nu = 2.0;  // degrees of freedom; lower = heavier tails

    return {
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

  // ─── 8. Neal's Funnel (true version — hierarchical) ─────────────────────────
  //
  //   v ~ N(0, 9),   x | v ~ N(0, exp(v))
  //
  //   The canonical funnel from Neal (2003). Here v = y (vertical axis),
  //   x = x (horizontal). Extremely challenging because the geometry changes
  //   drastically: wide at the top, narrow at the bottom.

  MCMC.targets['neals-funnel'] = (function () {
    return {
      logDensity: function (x, y) {
        // v = y, x = x
        var logp_v = -0.5 * y * y / 9;
        var var_x  = Math.exp(y);           // exp(v)
        var logp_x = -0.5 * x * x / var_x - 0.5 * y; // log N(x; 0, exp(v)) = -x²/(2 exp(v)) - v/2
        return logp_v + logp_x;
      },
      gradLogDensity: function (x, y) {
        var evy = Math.exp(y);
        var gx = -x / evy;
        // ∂/∂y: -y/9 + x²/(2 exp(y)) - 1/2
        var gy = -y / 9 + 0.5 * x * x / evy - 0.5;
        return [gx, gy];
      }
    };
  }());

  // ─── 9. Double Banana (figure-eight) ────────────────────────────────────────
  //
  //   Two back-to-back banana-shaped modes, useful for showing
  //   multi-modal + curved geometry simultaneously.

  MCMC.targets['double-banana'] = (function () {
    var b = 0.1, sigma = 0.5;

    function logComp(x, y, flip) {
      var y0  = flip * b * x * x;
      var dy  = y - y0;
      return -0.5 * x * x - 0.5 * dy * dy / (sigma * sigma);
    }
    function gradLogComp(x, y, flip) {
      var y0  = flip * b * x * x;
      var dy  = y - y0;
      var gx  = -x - dy * flip * 2 * b * x / (sigma * sigma);
      var gy  = dy / (sigma * sigma);
      return [gx, gy];
    }

    return {
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
