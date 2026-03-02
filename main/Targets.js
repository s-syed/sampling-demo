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
// 40 components, diagonal covariance, equal weights, scattered irregularly.
// Means ~ N(0, 6^2), scales ~ exp(Uniform[-0.5, 0.5]).

MCMC.targetNames.push('fab-gmm40');
MCMC.targets['fab-gmm40'] = (function() {
  var locs = [
    [1.058, 0.240], [0.587, 1.345], [1.121, -0.586], [0.570, -0.091],
    [-0.062, 0.246], [0.086, 0.873], [0.457, 0.073],  [0.266, 0.200],
    [0.896, -0.123], [0.188, -0.513], [-1.532, 0.392], [0.519, -0.445],
    [1.362, -0.873], [0.028, -0.112], [0.920, 0.882],  [0.093, 0.227],
    [-0.533,-1.189], [-0.209, 0.094], [0.738, 0.721],  [-0.232,-0.181],
    [-0.629,-0.852], [-1.024, 1.171], [-0.306,-0.263], [-0.752, 0.467],
    [-0.968,-0.128], [-0.537, 0.232], [-0.307,-0.708], [-0.017, 0.257],
    [0.040, 0.182],  [-0.381,-0.218], [-0.404,-0.216], [-0.488,-1.036],
    [0.107,-0.241],  [-0.978, 0.278], [-0.544, 0.031], [0.438, 0.077],
    [0.684,-0.741],  [0.241,-0.411],  [-0.523,-0.347], [-0.187, 0.034]
  ];
  var scales = [
    [0.139,0.061],[0.120,0.080],[0.127,0.159],[0.078,0.108],[0.110,0.108],
    [0.076,0.157],[0.095,0.141],[0.122,0.082],[0.137,0.090],[0.146,0.109],
    [0.147,0.121],[0.125,0.100],[0.158,0.116],[0.093,0.111],[0.062,0.082],
    [0.117,0.081],[0.113,0.093],[0.070,0.082],[0.107,0.110],[0.108,0.117],
    [0.116,0.093],[0.149,0.088],[0.094,0.148],[0.136,0.123],[0.067,0.152],
    [0.124,0.165],[0.070,0.145],[0.071,0.112],[0.069,0.142],[0.136,0.107],
    [0.091,0.065],[0.122,0.096],[0.125,0.144],[0.161,0.143],[0.061,0.087],
    [0.126,0.072],[0.102,0.064],[0.074,0.062],[0.134,0.076],[0.086,0.153]
  ];
  var K = locs.length;
  var logWk = Math.log(1/K);
  return {
    xmin: -3, xmax: 3,
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
  var R = 2.5, sigma2Ring = 0.20;
  var sigmaTimbit = 0.20;
  var logWRing = Math.log(0.60), logWTimbit = Math.log(0.40);

  function logRing(x) {
    var r = Math.sqrt(x[0]*x[0] + x[1]*x[1]);
    var d = r - R;
    return -d*d / (2*sigma2Ring);
  }
  function logTimbit(x) {
    return logMVNdiag2(x[0], x[1], 0, 0, sigmaTimbit, sigmaTimbit);
  }
  return {
    xmin: -5, xmax: 5,
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
