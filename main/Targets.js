/**
 * Targets.js — Extended target distributions for sampling-demo
 *
 * API (must match MCMC.js convention):
 *   logDensity(x)        — x is a vector; use x[0], x[1]
 *   gradLogDensity(x)    — returns matrix([[gx],[gy]])
 *   xmin, xmax           — viewing window (ymin/ymax derived symmetrically)
 *
 * Each target must also be pushed to MCMC.targetNames for the dropdown.
 */

// ─── Helpers ──────────────────────────────────────────────────────────────────

function logSumExpArr(arr) {
  var mx = -Infinity;
  for (var i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i];
  if (!isFinite(mx)) return -Infinity;
  var s = 0;
  for (var i = 0; i < arr.length; i++) s += Math.exp(arr[i] - mx);
  return mx + Math.log(s);
}

function logMVNdiag2(x0, x1, mu0, mu1, s0, s1) {
  var d0 = x0 - mu0, d1 = x1 - mu1;
  return -0.5 * (d0*d0/(s0*s0) + d1*d1/(s1*s1)) - Math.log(2*Math.PI*s0*s1);
}

// ─── 1. Multimodal Gaussian — 4 modes, equal weights ─────────────────────────

MCMC.targetNames.push('multimodal-gaussian');
MCMC.targets['multimodal-gaussian'] = (function() {
  var K = 4, spread = 3.0, sigma = 0.5;
  var modes = [];
  for (var k = 0; k < K; k++) {
    var angle = 2 * Math.PI * k / K;
    modes.push([spread * Math.cos(angle), spread * Math.sin(angle)]);
  }
  var logWk = Math.log(1/K);
  return {
    xmin: -6, xmax: 6,
    logDensity: function(x) {
      var terms = [];
      for (var k = 0; k < K; k++)
        terms.push(logWk + logMVNdiag2(x[0], x[1], modes[k][0], modes[k][1], sigma, sigma));
      return logSumExpArr(terms);
    },
    gradLogDensity: function(x) {
      var logTerms = [];
      for (var k = 0; k < K; k++)
        logTerms.push(logWk + logMVNdiag2(x[0], x[1], modes[k][0], modes[k][1], sigma, sigma));
      var lse = logSumExpArr(logTerms);
      var gx = 0, gy = 0;
      for (var k = 0; k < K; k++) {
        var r = Math.exp(logTerms[k] - lse);
        gx += r * (-(x[0] - modes[k][0]) / (sigma*sigma));
        gy += r * (-(x[1] - modes[k][1]) / (sigma*sigma));
      }
      return matrix([[gx],[gy]]);
    }
  };
}());

// ─── 2. Multimodal Gaussian — unequal weights ────────────────────────────────

MCMC.targetNames.push('multimodal-unequal');
MCMC.targets['multimodal-unequal'] = (function() {
  var comps = [
    { mu: [-4.0,  0.0], s: [0.6, 0.6], w: 0.5  },
    { mu: [ 4.0,  0.0], s: [0.4, 0.4], w: 0.3  },
    { mu: [ 0.0,  4.0], s: [0.5, 0.5], w: 0.15 },
    { mu: [ 0.0, -4.0], s: [0.3, 0.3], w: 0.05 },
  ];
  var logW = comps.map(function(c) { return Math.log(c.w); });
  return {
    xmin: -6, xmax: 6,
    logDensity: function(x) {
      var terms = comps.map(function(c, k) {
        return logW[k] + logMVNdiag2(x[0], x[1], c.mu[0], c.mu[1], c.s[0], c.s[1]);
      });
      return logSumExpArr(terms);
    },
    gradLogDensity: function(x) {
      var logTerms = comps.map(function(c, k) {
        return logW[k] + logMVNdiag2(x[0], x[1], c.mu[0], c.mu[1], c.s[0], c.s[1]);
      });
      var lse = logSumExpArr(logTerms);
      var gx = 0, gy = 0;
      for (var k = 0; k < comps.length; k++) {
        var r = Math.exp(logTerms[k] - lse);
        var c = comps[k];
        gx += r * (-(x[0] - c.mu[0]) / (c.s[0]*c.s[0]));
        gy += r * (-(x[1] - c.mu[1]) / (c.s[1]*c.s[1]));
      }
      return matrix([[gx],[gy]]);
    }
  };
}());

// ─── 3. GMM-32 (Blessing et al. arXiv:2208.01893) ────────────────────────────

MCMC.targetNames.push('gmm32');
MCMC.targets['gmm32'] = (function() {
  var modes = [];
  var rows = 4, cols = 8, spacing = 3.0, sigma = 0.3;
  for (var r = 0; r < rows; r++)
    for (var c = 0; c < cols; c++)
      modes.push([(c - (cols-1)/2)*spacing, (r - (rows-1)/2)*spacing]);
  var logWk = Math.log(1/modes.length);
  return {
    xmin: -14, xmax: 14,
    logDensity: function(x) {
      var terms = modes.map(function(m) {
        return logWk + logMVNdiag2(x[0], x[1], m[0], m[1], sigma, sigma);
      });
      return logSumExpArr(terms);
    },
    gradLogDensity: function(x) {
      var logTerms = modes.map(function(m) {
        return logWk + logMVNdiag2(x[0], x[1], m[0], m[1], sigma, sigma);
      });
      var lse = logSumExpArr(logTerms);
      var gx = 0, gy = 0;
      for (var k = 0; k < modes.length; k++) {
        var r = Math.exp(logTerms[k] - lse);
        gx += r * (-(x[0] - modes[k][0]) / (sigma*sigma));
        gy += r * (-(x[1] - modes[k][1]) / (sigma*sigma));
      }
      return matrix([[gx],[gy]]);
    }
  };
}());

// ─── 4. Ring / Donut ─────────────────────────────────────────────────────────

MCMC.targetNames.push('ring');
MCMC.targets['ring'] = (function() {
  var R = 3.0, sigma2 = 0.25;
  return {
    xmin: -6, xmax: 6,
    logDensity: function(x) {
      var r = Math.sqrt(x[0]*x[0] + x[1]*x[1]);
      var d = r - R;
      return -d*d / sigma2;
    },
    gradLogDensity: function(x) {
      var r = Math.sqrt(x[0]*x[0] + x[1]*x[1]);
      if (r < 1e-10) return matrix([[0],[0]]);
      var d = r - R;
      var scale = -2*d / (sigma2 * r);
      return matrix([[scale * x[0]], [scale * x[1]]]);
    }
  };
}());

// ─── 5. Rosenbrock ───────────────────────────────────────────────────────────

MCMC.targetNames.push('rosenbrock');
MCMC.targets['rosenbrock'] = {
  xmin: -3, xmax: 3,
  logDensity: function(x) {
    var a = 1 - x[0];
    var b = x[1] - x[0]*x[0];
    return -(a*a + 100*b*b) / 20.0;
  },
  gradLogDensity: function(x) {
    var b = x[1] - x[0]*x[0];
    var gx = (2*(1 - x[0]) + 400*x[0]*b) / 20.0;
    var gy = -200*b / 20.0;
    return matrix([[gx],[gy]]);
  }
};

// ─── 6. Bivariate Student-t (ν=2) ────────────────────────────────────────────

MCMC.targetNames.push('student-t');
MCMC.targets['student-t'] = {
  xmin: -8, xmax: 8,
  logDensity: function(x) {
    var nu = 2.0;
    var r2 = x[0]*x[0] + x[1]*x[1];
    return -0.5*(nu+2)*Math.log(1 + r2/nu);
  },
  gradLogDensity: function(x) {
    var nu = 2.0;
    var r2 = x[0]*x[0] + x[1]*x[1];
    var scale = -(nu+2)/(nu + r2);
    return matrix([[scale*x[0]],[scale*x[1]]]);
  }
};

// ─── 7. Neal's Funnel ────────────────────────────────────────────────────────

MCMC.targetNames.push('neals-funnel');
MCMC.targets['neals-funnel'] = {
  xmin: -8, xmax: 8,
  logDensity: function(x) {
    // x[0] = horizontal, x[1] = v (scale variable)
    var logp_v = -0.5*x[1]*x[1]/9;
    var ev = Math.exp(x[1]);
    var logp_x = -0.5*x[0]*x[0]/ev - 0.5*x[1];
    return logp_v + logp_x;
  },
  gradLogDensity: function(x) {
    var ev = Math.exp(x[1]);
    var gx = -x[0]/ev;
    var gy = -x[1]/9 + 0.5*x[0]*x[0]/ev - 0.5;
    return matrix([[gx],[gy]]);
  }
};

// ─── 8. Double Banana ────────────────────────────────────────────────────────

MCMC.targetNames.push('double-banana');
MCMC.targets['double-banana'] = (function() {
  var b = 0.1, sigma = 0.5;
  function logComp(x0, x1, flip) {
    var y0 = flip * b * x0*x0;
    var dy = x1 - y0;
    return -0.5*x0*x0 - 0.5*dy*dy/(sigma*sigma);
  }
  return {
    xmin: -5, xmax: 5,
    logDensity: function(x) {
      return logSumExpArr([
        logComp(x[0], x[1],  1) + Math.log(0.5),
        logComp(x[0], x[1], -1) + Math.log(0.5)
      ]);
    },
    gradLogDensity: function(x) {
      var t1 = logComp(x[0], x[1],  1) + Math.log(0.5);
      var t2 = logComp(x[0], x[1], -1) + Math.log(0.5);
      var lse = logSumExpArr([t1, t2]);
      var r1 = Math.exp(t1 - lse), r2 = Math.exp(t2 - lse);
      var gx = 0, gy = 0;
      [1, -1].forEach(function(flip, i) {
        var r = i === 0 ? r1 : r2;
        var y0 = flip * b * x[0]*x[0];
        var dy = x[1] - y0;
        gx += r * (-x[0] - dy * flip * 2*b*x[0] / (sigma*sigma));
        gy += r * (dy / (sigma*sigma));
      });
      return matrix([[gx],[gy]]);
    }
  };
}());
