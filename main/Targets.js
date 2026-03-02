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

// ─── 1. Multimodal Gaussian — unequal weights ────────────────────────────────

MCMC.targetNames.push('multimodal-unequal');
MCMC.targets['multimodal-unequal'] = (function() {
  var comps = [
    { mu: [-4.0,  0.0], s: [1.0, 1.0], w: 0.50 },
    { mu: [ 4.0,  0.0], s: [0.7, 0.7], w: 0.30 },
    { mu: [ 0.0,  3.5], s: [0.8, 0.8], w: 0.15 },
    { mu: [ 0.0, -3.5], s: [0.5, 0.5], w: 0.05 },
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
// 40 components, diagonal covariance, equal weights, scattered irregularly.
// Means ~ N(0, 6^2), scales ~ exp(Uniform[-0.5, 0.5]).

MCMC.targetNames.push('fab-gmm40');
MCMC.targets['fab-gmm40'] = (function() {
  var locs = [
    [5.292, 1.201], [2.936, 6.723], [5.603, -2.932], [2.851, -0.454],
    [-0.310, 1.232], [0.432, 4.363],  [2.283, 0.365],  [1.332, 1.001],
    [4.482, -0.616], [0.939, -2.563], [-7.659, 1.961], [2.594, -2.227],
    [6.810, -4.363],[0.138, -0.562], [4.599, 4.408],  [0.465, 1.135],
    [-2.664,-5.943],[-1.044, 0.469], [3.691, 3.607],  [-1.162,-0.907],
    [-3.146, -4.260],[-5.119, 5.853],[-1.529,-1.314], [-3.759, 2.333],
    [-4.842, -0.638],[-2.687, 1.161], [-1.533,-3.542], [-0.085, 1.285],
    [0.200, 0.908],  [-1.903,-1.088], [-2.018,-1.079], [-2.440,-5.179],
    [0.533, -1.206], [-4.891, 1.389], [-2.722, 0.156], [2.188, 0.387],
    [3.418, -3.705], [1.207, -2.055], [-2.613,-1.737], [-0.935, 0.169]
  ];
  var scales = [
    [1.389,0.609],[1.195,0.795],[1.265,1.588],[0.778,1.079],[1.096,1.075],
    [0.758,1.573],[0.948,1.414],[1.221,0.817],[1.369,0.902],[1.464,1.085],
    [1.465,1.212],[1.253,1.001],[1.578,1.155],[0.927,1.112],[0.618,0.820],
    [1.174,0.811],[1.125,0.931],[0.695,0.817],[1.072,1.095],[1.077,1.166],
    [1.164,0.934],[1.487,0.876],[0.938,1.480],[1.358,1.226],[0.670,1.521],
    [1.239,1.647],[0.704,1.445],[0.714,1.123],[0.686,1.416],[1.360,1.072],
    [0.911,0.650],[1.218,0.955],[1.249,1.443],[1.609,1.427],[0.614,0.869],
    [1.259,0.720],[1.021,0.640],[0.741,0.618],[1.341,0.759],[0.857,1.534]
  ];
  var K = locs.length;
  var logWk = Math.log(1/K);
  return {
    xmin: -10, xmax: 10,
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

// ─── 3. Donut + Timbit ───────────────────────────────────────────────────────
// Annular ring (75%) plus central Gaussian "timbit" (25%).
// Shows that even gradient-based methods can't tunnel between topological components.

MCMC.targetNames.push('donut-timbit');
MCMC.targets['donut-timbit'] = (function() {
  var R = 3.0, sigma2Ring = 0.30;
  var sigmaTimbit = 0.8;
  var logWRing = Math.log(0.75), logWTimbit = Math.log(0.25);

  function logRing(x) {
    var r = Math.sqrt(x[0]*x[0] + x[1]*x[1]);
    var d = r - R;
    return -d*d / (2*sigma2Ring);
  }
  function logTimbit(x) {
    return logMVNdiag2(x[0], x[1], 0, 0, sigmaTimbit, sigmaTimbit);
  }
  return {
    xmin: -7, xmax: 7,
    logDensity: function(x) {
      return logSumExpArr([logWRing + logRing(x), logWTimbit + logTimbit(x)]);
    },
    gradLogDensity: function(x) {
      var t1 = logWRing   + logRing(x);
      var t2 = logWTimbit + logTimbit(x);
      var lse = logSumExpArr([t1, t2]);
      var r1 = Math.exp(t1 - lse), r2 = Math.exp(t2 - lse);
      // Ring gradient
      var r = Math.sqrt(x[0]*x[0] + x[1]*x[1]);
      var gxRing = 0, gyRing = 0;
      if (r > 1e-10) {
        var d = r - R;
        var sc = -d / (sigma2Ring * r);
        gxRing = sc * x[0];
        gyRing = sc * x[1];
      }
      // Timbit gradient
      var gxTimbit = -x[0] / (sigmaTimbit*sigmaTimbit);
      var gyTimbit = -x[1] / (sigmaTimbit*sigmaTimbit);
      return matrix([[r1*gxRing + r2*gxTimbit],[r1*gyRing + r2*gyTimbit]]);
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
