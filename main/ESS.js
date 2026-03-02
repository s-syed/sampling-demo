/**
 * ESS.js — Effective Sample Size overlay for sampling-demo histograms
 *
 * Monkey-patches viz.drawHistograms (defined in Visualizer.js) to draw
 * ESS annotations on the x and y histogram canvases after each redraw.
 *
 * Must be loaded AFTER Visualizer.js in app.html.
 */

(function () {

  // ── ESS via initial monotone sequence estimator ──────────────────────────
  // Geyer (1992): sum pairs of autocorrelations while they stay positive.

  function computeESS(samples) {
    var n = samples.length;
    if (n < 4) return n;

    // mean
    var mean = 0;
    for (var i = 0; i < n; i++) mean += samples[i];
    mean /= n;

    // variance
    var v = 0;
    for (var i = 0; i < n; i++) { var d = samples[i] - mean; v += d * d; }
    v /= n;
    if (v < 1e-14) return n;

    // autocorrelations up to lag maxLag
    var maxLag = Math.min(Math.floor(n / 2), 200);
    var rho = [];
    for (var lag = 1; lag <= maxLag; lag++) {
      var s = 0;
      for (var i = 0; i < n - lag; i++)
        s += (samples[i] - mean) * (samples[i + lag] - mean);
      rho.push(s / (n * v));
    }

    // initial monotone sequence: sum pairs until pair sum goes negative
    var sumRho = 0;
    for (var k = 0; k < rho.length - 1; k += 2) {
      var pairSum = rho[k] + rho[k + 1];
      if (pairSum <= 0) break;
      sumRho += pairSum;
    }

    var ess = n / (1 + 2 * sumRho);
    return Math.max(1, Math.min(n, Math.round(ess)));
  }

  // ── Draw ESS label on a histogram canvas ─────────────────────────────────

  function drawESSLabel(canvas, ess, n, position) {
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;

    var pct = Math.round(ess / n * 100);
    var line1 = 'ESS: ' + ess + ' / ' + n;
    var line2 = 'Rel. ESS: ' + pct + '%';

    var fontSize = Math.max(18, Math.min(28, Math.floor(h * 0.12)));
    ctx.font = 'bold ' + fontSize + 'px sans-serif';
    ctx.textBaseline = 'top';

    var padding = 8;
    var lineGap = 6;
    var tw = Math.max(ctx.measureText(line1).width, ctx.measureText(line2).width);
    var boxW = tw + padding * 2;
    var boxH = fontSize * 2 + lineGap + padding * 2;

    var x, y;
    if (position === 'bottom-right') {
      x = w - boxW - padding;
      y = h - boxH - padding;
    } else {
      // top-left, offset down to avoid title overlap
      x = padding;
      y = padding + 28;
    }

    // background
    ctx.fillStyle = 'rgba(255,255,255,0.90)';
    ctx.fillRect(x, y, boxW, boxH);

    // text
    ctx.fillStyle = '#1a4a6e';
    ctx.fillText(line1, x + padding, y + padding);
    ctx.fillText(line2, x + padding, y + padding + fontSize + lineGap);
  }

  // ── Patch viz.drawHistograms ──────────────────────────────────────────────

  function installPatch() {
    if (typeof viz === 'undefined' || typeof sim === 'undefined') {
      // not ready yet — retry
      setTimeout(installPatch, 200);
      return;
    }
    if (!viz.drawHistograms) {
      setTimeout(installPatch, 200);
      return;
    }

    var _original = viz.drawHistograms.bind(viz);

    viz.drawHistograms = function () {
      _original();

      // only annotate if we have samples
      var chain = sim.mcmc && sim.mcmc.chain;
      if (!chain || chain.length < 2) return;

      var n = chain.length;

      // extract x and y marginals from chain
      // chain entries can be vectors (array-like) or matrix objects
      var xs = [], ys = [];
      for (var i = 0; i < n; i++) {
        var s = chain[i];
        if (s && typeof s[0] === 'number') {
          xs.push(s[0]);
          ys.push(s[1]);
        } else if (s && s.length >= 2) {
          xs.push(+s[0]);
          ys.push(+s[1]);
        }
      }

      if (xs.length < 4) return;

      var essX = computeESS(xs);
      var essY = computeESS(ys);

      drawESSLabel(viz.xHistCanvas, essX, xs.length, 'bottom-right');
      drawESSLabel(viz.yHistCanvas, essY, ys.length, 'top-left');
    };

    console.log('[ESS.js] drawHistograms patched.');
  }

  // Wait for window.onload to finish (viz and sim are created there)
  if (document.readyState === 'complete') {
    setTimeout(installPatch, 300);
  } else {
    window.addEventListener('load', function () { setTimeout(installPatch, 300); });
  }

})();
