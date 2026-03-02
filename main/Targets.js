/**
 * Targets.js — Extended target distributions for sampling-demo
 *
 * API (must match MCMC.js convention):
 *   logDensity(x)        — x is a vector; use x[0], x[1]
 *   gradLogDensity(x)    — returns matrix([[gx],[gy]])
 *   xmin, xmax           — viewing window (ymin/ymax derived symmetrically)
 *
 * Each new target must also be pushed to MCMC.targetNames for the dropdown.
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

// ─── Remove original 'multimodal' from dropdown ───────────────────────────────
// (It was registered in MCMC.js; splice it out before Simulation.js builds the UI)
(function() {
  var idx = MCMC.targetNames.indexOf('multimodal');
  if (idx !== -1) MCMC.targetNames.splice(idx, 1);
})();

// ─── 1a. Multimodal Gaussian — overlapping (wider modes) ─────────────────────

MCMC.targetNames.push('multimodal');
MCMC.targets['multimodal'] = (function() {
  var comps = [
    { mu: [-4.0,  0.0], s: [1.2, 1.2], w: 0.50 },
    { mu: [ 4.0,  0.0], s: [0.9, 0.9], w: 0.30 },
    { mu: [ 0.0,  3.5], s: [1.0, 1.0], w: 0.15 },
    { mu: [ 0.0, -3.5], s: [0.7, 0.7], w: 0.05 },
  ];
  var logW = comps.map(function(c) { return Math.log(c.w); });
  return {
    xmin: -8, xmax: 8,
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

// ─── 1b. Multimodal Gaussian — well-separated (sharp modes) ──────────────────

MCMC.targetNames.push('multimodal-well-separated');
MCMC.targets['multimodal-well-separated'] = (function() {
  var comps = [
    { mu: [-4.0,  0.0], s: [0.2, 0.2], w: 0.50 },
    { mu: [ 4.0,  0.0], s: [0.15, 0.15], w: 0.30 },
    { mu: [ 0.0,  3.5], s: [0.18, 0.18], w: 0.15 },
    { mu: [ 0.0, -3.5], s: [0.12, 0.12], w: 0.05 },
  ];
  var logW = comps.map(function(c) { return Math.log(c.w); });
  return {
    xmin: -8, xmax: 8,
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

// ─── 2. FAB GMM-40 (Midgley et al. / lollcat fab-torch, seed 0) ──────────────
// 40 components with multi-scale structure:
//   - 25 modes in a central cluster (some overlapping) spread over ~±4
//   - 10 modes in a mid-ring at radius ~7
//   - 5 isolated outlier modes at radius ~12
// Sigmas ~0.2–0.35 (peaky but visible). With proposal σ=0.1, mixing is poor.

MCMC.targetNames.push('fab-gmm40');
MCMC.targets['fab-gmm40'] = (function() {
  var locs = [
    // central dense cluster (25 modes, some overlapping)
    [ 0.0,  0.0], [ 0.6,  0.5], [-0.5,  0.7], [ 0.8, -0.6], [-0.7, -0.4],
    [ 1.5,  1.2], [-1.4,  1.0], [ 1.3, -1.5], [-1.6, -1.1], [ 2.2,  0.3],
    [-2.0,  0.8], [ 0.3,  2.1], [-0.4, -2.3], [ 2.5, -1.2], [-2.4, -1.8],
    [ 1.0,  2.8], [-1.2, -2.6], [ 3.0,  1.5], [-3.1,  0.5], [ 0.7, -3.2],
    [-0.8,  3.1], [ 3.2, -2.0], [-3.3, -1.5], [ 2.0,  3.2], [-2.2, -3.0],
    // mid-ring (10 modes at radius ~7)
    [ 7.0,  0.0], [-7.0,  0.0], [ 0.0,  7.0], [ 0.0, -7.0],
    [ 5.0,  5.0], [-5.0,  5.0], [ 5.0, -5.0], [-5.0, -5.0],
    [ 6.5,  3.0], [-6.5, -3.0],
    // outliers (5 modes at radius ~12)
    [12.0,  0.0], [-12.0,  0.0], [ 0.0, 12.0], [ 8.5,  8.5], [-9.0, -8.0]
  ];
  var scales = [
    // central cluster: slightly varied sigmas, some bigger for overlap
    [0.30,0.30],[0.35,0.25],[0.28,0.32],[0.22,0.28],[0.30,0.22],
    [0.25,0.30],[0.28,0.25],[0.22,0.30],[0.30,0.28],[0.25,0.22],
    [0.28,0.30],[0.30,0.25],[0.22,0.28],[0.25,0.30],[0.30,0.22],
    [0.28,0.25],[0.25,0.28],[0.22,0.25],[0.28,0.22],[0.25,0.30],
    [0.30,0.28],[0.22,0.30],[0.28,0.25],[0.25,0.22],[0.30,0.28],
    // mid-ring: slightly peakier
    [0.22,0.22],[0.22,0.22],[0.22,0.22],[0.22,0.22],
    [0.20,0.20],[0.20,0.20],[0.20,0.20],[0.20,0.20],
    [0.20,0.22],[0.22,0.20],
    // outliers: sharpest
    [0.18,0.18],[0.18,0.18],[0.18,0.18],[0.18,0.18],[0.18,0.18]
  ];
  var K = locs.length;
  var logWk = Math.log(1/K);
  return {
    xmin: -15, xmax: 15,
    logDensity: function(x) {
      var terms = [];
      for (var k = 0; k < K; k++)
        terms.push(logWk + logMVNdiag2(x[0], x[1], locs[k][0], locs[k][1], scales[k][0], scales[k][1]));
      return logSumExpArr(terms);
    },
    gradLogDensity: function(x) {
      var logTerms = [];
      for (var k = 0; k < K; k++)
        logTerms.push(logWk + logMVNdiag2(x[0], x[1], locs[k][0], locs[k][1], scales[k][0], scales[k][1]));
      var lse = logSumExpArr(logTerms);
      var gx = 0, gy = 0;
      for (var k = 0; k < K; k++) {
        var r = Math.exp(logTerms[k] - lse);
        gx += r * (-(x[0] - locs[k][0]) / (scales[k][0]*scales[k][0]));
        gy += r * (-(x[1] - locs[k][1]) / (scales[k][1]*scales[k][1]));
      }
      return matrix([[gx],[gy]]);
    }
  };
}());

// ─── 3. Twin Donuts ──────────────────────────────────────────────────────────
// Two annular rings well-separated, equal weights.
// Great for showing that samplers get trapped in one ring.

MCMC.targetNames.push('twin-donuts');
MCMC.targets['twin-donuts'] = (function() {
  var R = 1.5, sigma2 = 0.06;
  var cx = [-3.0, 3.0]; // centres of the two rings
  var logWk = Math.log(0.5);

  function logRing(x, cx0) {
    var dx = x[0] - cx0;
    var r = Math.sqrt(dx*dx + x[1]*x[1]);
    var d = r - R;
    return -d*d / (2*sigma2);
  }
  return {
    xmin: -7, xmax: 7,
    logDensity: function(x) {
      return logSumExpArr([logWk + logRing(x, cx[0]), logWk + logRing(x, cx[1])]);
    },
    gradLogDensity: function(x) {
      var t1 = logWk + logRing(x, cx[0]);
      var t2 = logWk + logRing(x, cx[1]);
      var lse = logSumExpArr([t1, t2]);
      var r1 = Math.exp(t1 - lse), r2 = Math.exp(t2 - lse);
      var gx = 0, gy = 0;
      [cx[0], cx[1]].forEach(function(cx0, i) {
        var w = i === 0 ? r1 : r2;
        var dx = x[0] - cx0;
        var r = Math.sqrt(dx*dx + x[1]*x[1]);
        if (r < 1e-10) return;
        var d = r - R;
        var sc = -d / (sigma2 * r);
        gx += w * sc * dx;
        gy += w * sc * x[1];
      });
      return matrix([[gx],[gy]]);
    }
  };
}());

// ─── 4. Rosenbrock ───────────────────────────────────────────────────────────

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

// ─── 5. Bivariate Student-t (nu=2) ───────────────────────────────────────────

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

// ─── 6. Neal's Funnel ────────────────────────────────────────────────────────

MCMC.targetNames.push('neals-funnel');
MCMC.targets['neals-funnel'] = {
  xmin: -8, xmax: 8,
  logDensity: function(x) {
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
