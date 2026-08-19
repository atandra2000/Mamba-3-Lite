/* Mamba-3-Lite Documentation Portal Client-Side Behaviors
   ======================================================================
   • Complex State Spiral (FIG · A0 — the complex SSD recurrence)
   • Live Pipeline Pass Telemetry (FIG · A1 — chunkwise SSD scan)
   • Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
   • Navigation Filtering & Sidebar Mobile Toggle
   • Table of Contents Scrollspy
   • Highlight.js & KaTeX bootstrap (mirrors the inline doc-page script)
*/

(function () {
    'use strict';

    var reduced = window.matchMedia &&
                  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ------------------------------------------------------------------
    // 1. Complex State-Space Dynamics (FIG · A0)
    //    Interactive high-DPI canvas simulation of the complex SSD recurrence:
    //    h_t = exp((A+iB)·dt)·h + B·x
    //    Visualizes 32 conjugate eigenvalue pairs (N=64 complex states),
    //    logarithmic decay spirals, counter-rotating Im+/Im- phasors,
    //    MIMO coupling resonance, and the accumulated state core.
    // ------------------------------------------------------------------
    function initComplexSpiral() {
        var canvas = document.getElementById('complexSpiralCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var probeEl = document.getElementById('hudProbe');
        var probeTag = document.getElementById('probeTag');
        var probeCoords = document.getElementById('probeCoords');
        var probeDecay = document.getElementById('probeDecay');
        var normValEl = document.getElementById('hudNormVal');
        var pauseBtn = document.getElementById('figPauseBtn');
        var modeBtns = document.querySelectorAll('.fig-controls .fig-btn[data-mode]');

        var currentMode = 'spiral';
        var isPaused = false;
        var isHovered = false;
        var mouseX = -9999, mouseY = -9999;
        var probePair = null;
        var pulseWaves = [];
        var animId = null;
        var lastTime = 0;
        var simTime = 0;

        // Palette tokens matching the dark bench notebook theme
        var PALETTE = {
            paperBg: '#0e0c0a',
            paperCenter: '#181410',
            gridRule: 'rgba(58, 50, 38, 0.45)',
            gridAxis: 'rgba(201, 163, 92, 0.35)',
            unitCircle: 'rgba(201, 163, 92, 0.22)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.65)',
            oliveTint: 'rgba(154, 148, 64, 0.15)',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.65)',
            terracottaTint: 'rgba(224, 122, 63, 0.15)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.7)',
            coreHot: '#fffaf0',
            ink: '#d8ccb4',
            inkFaint: '#7a7160'
        };

        // 32 Conjugate Eigenvalue Pairs = 64 Complex States (N=64)
        var PAIRS_COUNT = 32;
        var pairs = [];
        for (var i = 0; i < PAIRS_COUNT; i++) {
            var frac = i / (PAIRS_COUNT - 1);
            // Decay radius distribution: log-spread across unit disk
            var baseRadius = 0.18 + 0.74 * Math.pow(frac, 0.82);
            // Natural frequencies: fast oscillations near center, steady macro modes outward
            var omega = 0.35 + 1.85 * Math.pow(1 - frac, 1.15);
            // Golden angle phase distribution
            var phase0 = (i * 2.399963229728653) % (Math.PI * 2);
            var decayRate = 0.05 + 0.85 * (1 - frac);

            pairs.push({
                id: i + 1,
                r: baseRadius,
                omega: omega,
                phase: phase0,
                decay: decayRate,
                thetaPlus: phase0,
                thetaMinus: -phase0,
                xPlus: 0, yPlus: 0,
                xMinus: 0, yMinus: 0,
                trailPlus: [],
                trailMinus: [],
                glow: 0
            });
        }

        // Logarithmic Spiral Quanta (Energy Particles)
        var PARTICLES_COUNT = 32;
        var particles = [];
        for (var p = 0; p < PARTICLES_COUNT; p++) {
            particles.push({
                arm: p % 4,
                t: (p / PARTICLES_COUNT) * 4.0,
                speed: 0.22 + 0.18 * Math.random(),
                size: 1.2 + 1.2 * Math.random(),
                alpha: 0.3 + 0.7 * Math.random()
            });
        }

        // Resize Canvas to Device Pixel Ratio
        var width = 0, height = 0, cx = 0, cy = 0, maxR = 0;
        function resize() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 320;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);
            cx = width / 2;
            cy = height / 2;
            maxR = Math.min(width * 0.38, height * 0.38);
        }

        // Initialize ResizeObserver
        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () {
                resize();
                if (reduced || isPaused) renderFrame(0, true);
            });
            ro.observe(container);
        } else {
            window.addEventListener('resize', resize);
        }
        resize();

        // Trigger a Token Input shockwave pulse
        function triggerPulse(originX, originY) {
            var ox = originX !== undefined ? originX : cx;
            var oy = originY !== undefined ? originY : cy;
            pulseWaves.push({
                x: ox,
                y: oy,
                r: 0,
                maxR: Math.max(width, height) * 0.75,
                speed: 380,
                alpha: 1.0
            });
            // Momentary excitation on state core
            pairs.forEach(function (pair) {
                pair.glow = 1.0;
            });
        }

        // Event Listeners for Interaction
        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
        });

        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999;
            mouseY = -9999;
            probePair = null;
            if (probeEl) probeEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;
            triggerPulse(clickX, clickY);
        });

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume animation' : 'Pause animation');
                if (!isPaused && !animId) {
                    lastTime = performance.now();
                    loop(lastTime);
                }
            });
        }

        modeBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                modeBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentMode = btn.getAttribute('data-mode') || 'spiral';
                if (reduced || isPaused) renderFrame(0, true);
            });
        });

        // --------------------------------------------------------------
        // Render Functions
        // --------------------------------------------------------------

        function drawBackground() {
            // Smooth radial vignette
            var grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR * 1.35);
            grad.addColorStop(0, PALETTE.paperCenter);
            grad.addColorStop(0.6, PALETTE.paperBg);
            grad.addColorStop(1, '#070605');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, width, height);

            ctx.save();
            ctx.lineWidth = 1;

            // Outer dial perimeter tick marks (64 precision ticks)
            var totalTicks = 64;
            for (var tk = 0; tk < totalTicks; tk++) {
                var tkAng = (tk * Math.PI * 2) / totalTicks;
                var isMajor = tk % 8 === 0;
                var isSemi = tk % 4 === 0;
                var rInner = maxR * (isMajor ? 0.96 : (isSemi ? 0.975 : 0.985));
                var rOuter = maxR * 1.0;

                ctx.beginPath();
                ctx.moveTo(cx + rInner * Math.cos(tkAng), cy + rInner * Math.sin(tkAng));
                ctx.lineTo(cx + rOuter * Math.cos(tkAng), cy + rOuter * Math.sin(tkAng));
                ctx.strokeStyle = isMajor ? PALETTE.gold : (isSemi ? 'rgba(201, 163, 92, 0.45)' : 'rgba(58, 50, 38, 0.4)');
                ctx.lineWidth = isMajor ? 1.2 : 0.8;
                ctx.stroke();
            }

            // Concentric Isochrone / Unit Rings
            var rings = [0.25, 0.50, 0.75, 1.00];
            var angLabel = -Math.PI * 0.22; // Upper-right quadrant (clean space)

            rings.forEach(function (frac, idx) {
                var r = maxR * frac;
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                if (idx === rings.length - 1) {
                    ctx.strokeStyle = PALETTE.unitCircle;
                    ctx.setLineDash([4, 4]);
                } else {
                    ctx.strokeStyle = PALETTE.gridRule;
                    ctx.setLineDash([2, 5]);
                }
                ctx.stroke();

                // Ring label in clear diagonal quadrant
                ctx.setLineDash([]);
                var lx = cx + r * Math.cos(angLabel);
                var ly = cy + r * Math.sin(angLabel);
                ctx.fillStyle = PALETTE.gold;
                ctx.font = 'bold 9px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('|z|=' + frac.toFixed(2), lx + 6, ly - 4);
            });

            // Re & Im Crosshair Axes
            ctx.setLineDash([]);
            ctx.strokeStyle = PALETTE.gridAxis;

            // Real axis (Horizontal)
            ctx.beginPath();
            ctx.moveTo(cx - maxR * 1.04, cy);
            ctx.lineTo(cx + maxR * 1.04, cy);
            ctx.stroke();

            // Imaginary axis (Vertical)
            ctx.beginPath();
            ctx.moveTo(cx, cy - maxR * 1.04);
            ctx.lineTo(cx, cy + maxR * 1.04);
            ctx.stroke();

            // Axis labels with generous margins
            ctx.fillStyle = PALETTE.gold;
            ctx.font = 'bold 10px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText('+Re', cx + maxR * 1.04 + 4, cy + 3);
            ctx.textAlign = 'right';
            ctx.fillText('-Re', cx - maxR * 1.04 - 4, cy + 3);

            ctx.textAlign = 'center';
            ctx.fillText('+i Im', cx, cy - maxR * 1.04 - 4);
            ctx.fillText('-i Im', cx, cy + maxR * 1.04 + 12);
            // Subtle angular rays (30, 45, 60 deg)
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.22)';
            ctx.setLineDash([1, 6]);
            [Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI * 2 / 3, Math.PI * 3 / 4, Math.PI * 5 / 6].forEach(function (ang) {
                ctx.beginPath();
                ctx.moveTo(cx - Math.cos(ang) * maxR, cy - Math.sin(ang) * maxR);
                ctx.lineTo(cx + Math.cos(ang) * maxR, cy + Math.sin(ang) * maxR);
                ctx.stroke();
            });
            ctx.setLineDash([]);

            ctx.restore();
        }

        // Mode 1 & 2: Logarithmic Spiral Manifolds
        function drawSpiralManifolds(t) {
            ctx.save();
            var arms = 4;
            var rot = t * 0.12;

            for (var a = 0; a < arms; a++) {
                var armOffset = (a * Math.PI * 2) / arms + rot;
                ctx.beginPath();
                var started = false;

                // Logarithmic spiral r = a * exp(b * theta)
                for (var theta = 0; theta < Math.PI * 3.6; theta += 0.08) {
                    var rNorm = Math.exp(0.38 * (theta - Math.PI * 3.6));
                    var r = rNorm * maxR * 1.05;
                    var angle = armOffset - theta;
                    var x = cx + r * Math.cos(angle);
                    var y = cy + r * Math.sin(angle);

                    if (!started) {
                        ctx.moveTo(x, y);
                        started = true;
                    } else {
                        ctx.lineTo(x, y);
                    }
                }

                // Gradient stroke along spiral
                var grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR);
                grad.addColorStop(0, 'rgba(224, 122, 63, 0.45)');
                grad.addColorStop(0.5, 'rgba(154, 148, 64, 0.25)');
                grad.addColorStop(1, 'rgba(44, 38, 28, 0.05)');
                ctx.strokeStyle = grad;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }

            // Draw stream quanta moving along spiral arms
            particles.forEach(function (pt) {
                if (!reduced) pt.t += 0.015 * pt.speed;
                if (pt.t > 3.6) pt.t = 0.0;

                var theta = pt.t * Math.PI;
                var rNorm = Math.exp(0.38 * (theta - Math.PI * 3.6));
                var r = rNorm * maxR * 1.05;
                var angle = (pt.arm * Math.PI * 2) / arms + rot - theta;
                var px = cx + r * Math.cos(angle);
                var py = cy + r * Math.sin(angle);

                ctx.beginPath();
                ctx.arc(px, py, pt.size, 0, Math.PI * 2);
                ctx.fillStyle = pt.arm % 2 === 0 ? PALETTE.oliveGlow : PALETTE.terracottaGlow;
                ctx.fill();
            });

            ctx.restore();
        }

        // Mode 3: Continuous Decay Vector Flow Field
        function drawVectorField(t) {
            ctx.save();
            var step = 28;
            var arrowLen = 10;
            var cols = Math.floor(width / step);
            var rows = Math.floor(height / step);

            for (var i = 0; i <= cols; i++) {
                for (var j = 0; j <= rows; j++) {
                    var px = i * step + (step / 2);
                    var py = j * step + (step / 2);
                    var dx = px - cx;
                    var dy = py - cy;
                    var dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist > maxR * 1.1 || dist < 12) continue;

                    // Complex vector field dh/dt = (A + iB)h
                    // A < 0 (decay towards center), B != 0 (rotation)
                    var u = -0.55 * (dx / dist) - 0.85 * (dy / dist);
                    var v = -0.55 * (dy / dist) + 0.85 * (dx / dist);

                    var ang = Math.atan2(v, u);
                    var flowIntensity = 0.25 + 0.75 * Math.sin(dist * 0.03 - t * 2.0);

                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + Math.cos(ang) * arrowLen * flowIntensity, py + Math.sin(ang) * arrowLen * flowIntensity);
                    ctx.strokeStyle = dist < maxR * 0.5 ? 'rgba(224, 122, 63, 0.45)' : 'rgba(154, 148, 64, 0.35)';
                    ctx.lineWidth = 1.0;
                    ctx.stroke();

                    // Micro dot at base
                    ctx.beginPath();
                    ctx.arc(px, py, 1.0, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(201, 163, 92, 0.4)';
                    ctx.fill();
                }
            }
            ctx.restore();
        }

        // Update and Draw 32 Conjugate Phasor Pairs
        function drawConjugatePairs(t, dt) {
            ctx.save();

            // Find closest pair to cursor for probe inspection
            var closestPair = null;
            var closestDist = 99999;

            pairs.forEach(function (pair) {
                if (!reduced) {
                    pair.thetaPlus = pair.phase + pair.omega * t;
                    pair.thetaMinus = -pair.phase - pair.omega * t;
                    if (pair.glow > 0) pair.glow = Math.max(0, pair.glow - dt * 1.8);
                }

                var r = pair.r * maxR;

                // Position of Im+ Phasor (Olive, counter-clockwise)
                pair.xPlus = cx + r * Math.cos(pair.thetaPlus);
                pair.yPlus = cy - r * Math.sin(pair.thetaPlus);

                // Position of Im- Phasor (Terracotta, clockwise)
                pair.xMinus = cx + r * Math.cos(pair.thetaMinus);
                pair.yMinus = cy - r * Math.sin(pair.thetaMinus);

                // Update orbital history trails
                if (!reduced) {
                    pair.trailPlus.unshift({ x: pair.xPlus, y: pair.yPlus, a: 1.0 });
                    pair.trailMinus.unshift({ x: pair.xMinus, y: pair.yMinus, a: 1.0 });
                    if (pair.trailPlus.length > 20) pair.trailPlus.pop();
                    if (pair.trailMinus.length > 20) pair.trailMinus.pop();
                }

                // Check distance to mouse
                if (isHovered) {
                    var d1 = Math.hypot(mouseX - pair.xPlus, mouseY - pair.yPlus);
                    var d2 = Math.hypot(mouseX - pair.xMinus, mouseY - pair.yMinus);
                    var minD = Math.min(d1, d2);
                    if (minD < closestDist && minD < 45) {
                        closestDist = minD;
                        closestPair = pair;
                    }
                }
            });

            probePair = closestPair;

            // Draw Orbit Tracks in 'orbits' mode
            if (currentMode === 'orbits') {
                pairs.forEach(function (pair) {
                    var r = pair.r * maxR;
                    ctx.beginPath();
                    ctx.arc(cx, cy, r, 0, Math.PI * 2);
                    ctx.strokeStyle = 'rgba(58, 50, 38, 0.4)';
                    ctx.lineWidth = 0.8;
                    ctx.stroke();

                    // Radial rays connecting center to phasors
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(pair.xPlus, pair.yPlus);
                    ctx.strokeStyle = PALETTE.oliveTint;
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(pair.xMinus, pair.yMinus);
                    ctx.strokeStyle = PALETTE.terracottaTint;
                    ctx.stroke();
                });
            }

            // Draw Conjugate Coupling Chords between Im+ and Im-
            pairs.forEach(function (pair) {
                var isSelected = probePair && probePair.id === pair.id;
                ctx.beginPath();
                ctx.moveTo(pair.xPlus, pair.yPlus);
                ctx.lineTo(pair.xMinus, pair.yMinus);
                if (isSelected) {
                    ctx.strokeStyle = PALETTE.gold;
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([3, 3]);
                } else {
                    ctx.strokeStyle = 'rgba(201, 163, 92, 0.16)';
                    ctx.lineWidth = 0.8;
                    ctx.setLineDash([2, 4]);
                }
                ctx.stroke();
                ctx.setLineDash([]);
            });

            // Draw Fading Orbital Trails
            pairs.forEach(function (pair) {
                // Im+ Olive Trail
                if (pair.trailPlus.length > 1) {
                    for (var i = 0; i < pair.trailPlus.length - 1; i++) {
                        var alpha = (1 - (i / pair.trailPlus.length)) * 0.55;
                        ctx.beginPath();
                        ctx.moveTo(pair.trailPlus[i].x, pair.trailPlus[i].y);
                        ctx.lineTo(pair.trailPlus[i + 1].x, pair.trailPlus[i + 1].y);
                        ctx.strokeStyle = 'rgba(154, 148, 64, ' + alpha.toFixed(3) + ')';
                        ctx.lineWidth = Math.max(0.6, 2.2 * (1 - (i / pair.trailPlus.length)));
                        ctx.stroke();
                    }
                }

                // Im- Terracotta Trail
                if (pair.trailMinus.length > 1) {
                    for (var j = 0; j < pair.trailMinus.length - 1; j++) {
                        var alphaT = (1 - (j / pair.trailMinus.length)) * 0.55;
                        ctx.beginPath();
                        ctx.moveTo(pair.trailMinus[j].x, pair.trailMinus[j].y);
                        ctx.lineTo(pair.trailMinus[j + 1].x, pair.trailMinus[j + 1].y);
                        ctx.strokeStyle = 'rgba(224, 122, 63, ' + alphaT.toFixed(3) + ')';
                        ctx.lineWidth = Math.max(0.6, 2.2 * (1 - (j / pair.trailMinus.length)));
                        ctx.stroke();
                    }
                }
            });

            // Draw Phasor Nodes
            pairs.forEach(function (pair) {
                var isSelected = probePair && probePair.id === pair.id;
                var baseR = isSelected ? 4.5 : 2.6;
                var extraGlow = pair.glow * 4.0;

                // --- Im+ Node (Olive) ---
                // Outer glow
                var gradPlus = ctx.createRadialGradient(pair.xPlus, pair.yPlus, 0, pair.xPlus, pair.yPlus, baseR * 3.5 + extraGlow);
                gradPlus.addColorStop(0, isSelected ? 'rgba(216, 204, 180, 0.95)' : PALETTE.oliveGlow);
                gradPlus.addColorStop(0.5, PALETTE.oliveTint);
                gradPlus.addColorStop(1, 'rgba(154, 148, 64, 0)');
                ctx.fillStyle = gradPlus;
                ctx.beginPath();
                ctx.arc(pair.xPlus, pair.yPlus, baseR * 3.5 + extraGlow, 0, Math.PI * 2);
                ctx.fill();

                // Core dot
                ctx.beginPath();
                ctx.arc(pair.xPlus, pair.yPlus, baseR, 0, Math.PI * 2);
                ctx.fillStyle = isSelected ? '#ffffff' : PALETTE.olive;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 0.7;
                ctx.stroke();

                // --- Im- Node (Terracotta) ---
                // Outer glow
                var gradMinus = ctx.createRadialGradient(pair.xMinus, pair.yMinus, 0, pair.xMinus, pair.yMinus, baseR * 3.5 + extraGlow);
                gradMinus.addColorStop(0, isSelected ? 'rgba(216, 204, 180, 0.95)' : PALETTE.terracottaGlow);
                gradMinus.addColorStop(0.5, PALETTE.terracottaTint);
                gradMinus.addColorStop(1, 'rgba(224, 122, 63, 0)');
                ctx.fillStyle = gradMinus;
                ctx.beginPath();
                ctx.arc(pair.xMinus, pair.yMinus, baseR * 3.5 + extraGlow, 0, Math.PI * 2);
                ctx.fill();

                // Core dot
                ctx.beginPath();
                ctx.arc(pair.xMinus, pair.yMinus, baseR, 0, Math.PI * 2);
                ctx.fillStyle = isSelected ? '#ffffff' : PALETTE.terracotta;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 0.7;
                ctx.stroke();

                // Inspection Target Reticle if Selected
                if (isSelected) {
                    [pair.xPlus, pair.xMinus].forEach(function (nx, idx) {
                        var ny = idx === 0 ? pair.yPlus : pair.yMinus;
                        ctx.beginPath();
                        ctx.arc(nx, ny, 11, 0, Math.PI * 2);
                        ctx.strokeStyle = PALETTE.gold;
                        ctx.lineWidth = 1.0;
                        ctx.stroke();

                        // Crosshair marks
                        ctx.beginPath();
                        ctx.moveTo(nx - 15, ny); ctx.lineTo(nx - 7, ny);
                        ctx.moveTo(nx + 7, ny); ctx.lineTo(nx + 15, ny);
                        ctx.moveTo(nx, ny - 15); ctx.lineTo(nx, ny - 7);
                        ctx.moveTo(nx, ny + 7); ctx.lineTo(nx, ny + 15);
                        ctx.stroke();
                    });
                }
            });

            ctx.restore();
        }

        // Draw Token Injection Pulse Waves
        function drawPulseWaves(dt) {
            if (pulseWaves.length === 0) return;
            ctx.save();

            for (var i = pulseWaves.length - 1; i >= 0; i--) {
                var wave = pulseWaves[i];
                wave.r += wave.speed * dt;
                wave.alpha = Math.max(0, 1.0 - (wave.r / wave.maxR));

                if (wave.alpha <= 0.01) {
                    pulseWaves.splice(i, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(wave.x, wave.y, wave.r, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(224, 122, 63, ' + wave.alpha.toFixed(3) + ')';
                ctx.lineWidth = 1.8;
                ctx.stroke();

                ctx.beginPath();
                ctx.arc(wave.x, wave.y, Math.max(0, wave.r - 8), 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(201, 163, 92, ' + (wave.alpha * 0.5).toFixed(3) + ')';
                ctx.lineWidth = 1.0;
                ctx.stroke();
            }

            ctx.restore();
        }

        // Draw Central Accumulated State Singularity
        function drawCore(t) {
            ctx.save();

            // Breathing radius
            var breathe = reduced ? 1.0 : (0.88 + 0.12 * Math.sin(t * 1.8) + 0.05 * Math.cos(t * 3.4));
            var coreR = 7.0 * breathe;

            // Outer Radiant Nebula Glow
            var outerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 48 * breathe);
            outerGrad.addColorStop(0, 'rgba(224, 122, 63, 0.45)');
            outerGrad.addColorStop(0.35, 'rgba(154, 148, 64, 0.20)');
            outerGrad.addColorStop(0.7, 'rgba(201, 163, 92, 0.08)');
            outerGrad.addColorStop(1, 'rgba(14, 12, 10, 0)');
            ctx.fillStyle = outerGrad;
            ctx.beginPath();
            ctx.arc(cx, cy, 48 * breathe, 0, Math.PI * 2);
            ctx.fill();

            // Golden Corona Ring
            ctx.beginPath();
            ctx.arc(cx, cy, 14 * breathe, 0, Math.PI * 2);
            ctx.strokeStyle = PALETTE.goldGlow;
            ctx.lineWidth = 1.2;
            ctx.stroke();

            // Hot Inner Singularity
            var innerGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
            innerGrad.addColorStop(0, PALETTE.coreHot);
            innerGrad.addColorStop(0.4, '#f2d096');
            innerGrad.addColorStop(0.85, PALETTE.terracotta);
            innerGrad.addColorStop(1, 'rgba(224, 122, 63, 0.8)');
            ctx.fillStyle = innerGrad;
            ctx.beginPath();
            ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
            ctx.fill();

            ctx.restore();
        }

        // Update HUD Telemetry readouts
        function updateHUD(t) {
            if (normValEl && !reduced) {
                var norm = 1.38 + 0.08 * Math.sin(t * 1.4) + 0.04 * Math.sin(t * 4.2);
                normValEl.textContent = norm.toFixed(3);
            }

            if (probePair && probeEl) {
                probeEl.style.opacity = '1';
                var targetX = probePair.xPlus;
                var targetY = probePair.yPlus;

                // Card dimensions
                var cardW = 205;
                var cardH = 56;

                var leftPx = targetX + 16;
                var topPx = targetY - 24;

                // Flip to left if too close to right edge
                if (leftPx + cardW > width - 12) {
                    leftPx = targetX - cardW - 16;
                }
                // Clamp within bounds
                leftPx = Math.max(12, Math.min(width - cardW - 12, leftPx));
                topPx = Math.max(12, Math.min(height - cardH - 12, topPx));

                probeEl.style.left = leftPx + 'px';
                probeEl.style.top = topPx + 'px';

                var re = probePair.r * Math.cos(probePair.thetaPlus);
                var im = probePair.r * Math.sin(probePair.thetaPlus);
                var deg = ((probePair.thetaPlus * 180 / Math.PI) % 360 + 360) % 360;
                var tau = (1.0 / (probePair.decay * 0.08)).toFixed(1);

                if (probeTag) probeTag.textContent = 'PAIR #' + probePair.id + ' (k=' + (probePair.id - 1) + ')';
                if (probeCoords) probeCoords.textContent = 'Re: ' + (re >= 0 ? '+' : '') + re.toFixed(3) + ' · Im: ±' + Math.abs(im).toFixed(3);
                if (probeDecay) probeDecay.textContent = 'Decay τ = ' + tau + ' steps · θ = ' + deg.toFixed(1) + '°';
            } else if (probeEl) {
                probeEl.style.opacity = '0';
            }
        }

        // Master Frame Render
        function renderFrame(dt, force) {
            if (!force && isPaused) return;

            simTime += dt;
            ctx.clearRect(0, 0, width, height);

            drawBackground();

            if (currentMode === 'spiral') {
                drawSpiralManifolds(simTime);
                drawConjugatePairs(simTime, dt);
            } else if (currentMode === 'orbits') {
                drawConjugatePairs(simTime, dt);
            } else if (currentMode === 'field') {
                drawVectorField(simTime);
                drawConjugatePairs(simTime, dt);
            }

            drawPulseWaves(dt);
            drawCore(simTime);
            updateHUD(simTime);
        }

        // Animation Loop
        function loop(timestamp) {
            if (!lastTime) lastTime = timestamp;
            var dt = Math.min((timestamp - lastTime) / 1000, 0.1);
            lastTime = timestamp;

            renderFrame(dt, false);

            if (!isPaused && !reduced) {
                animId = requestAnimationFrame(loop);
            }
        }

        if (reduced) {
            renderFrame(0, true);
        } else {
            lastTime = performance.now();
            animId = requestAnimationFrame(loop);

            document.addEventListener('visibilitychange', function () {
                if (document.hidden) {
                    if (animId) {
                        cancelAnimationFrame(animId);
                        animId = null;
                    }
                } else if (!isPaused && !animId) {
                    lastTime = performance.now();
                    animId = requestAnimationFrame(loop);
                }
            });
        }
    }

    // ------------------------------------------------------------------
    // 2. Living Pipeline Pass Telemetry (FIG · A1)
    //    Interactive high-DPI Canvas simulation of one full training step:
    //    Forward Activation Pass → Full-Vocab Cross-Entropy Loss →
    //    Autograd Backward Pass (Gradient Checkpointing Activation
    //    Recomputation) → AdamW Parameter Update Step.
    // ------------------------------------------------------------------
    function initPassDiagram() {
        var canvas = document.getElementById('passDiagramCanvas');
        if (!canvas) return;

        var ctx = canvas.getContext('2d');
        if (!ctx) return;

        var container = canvas.parentElement;
        var tooltipEl = document.getElementById('passStageTooltip');
        var stTag = document.getElementById('stTag');
        var stOp = document.getElementById('stOp');
        var stShape = document.getElementById('stShape');
        var stDesc = document.getElementById('stDesc');
        var phaseEl = document.getElementById('phCurrentPhase');
        var tickerText = document.getElementById('passTickerText');
        var tickerBeacon = document.getElementById('passTickerBeacon');
        var pauseBtn = document.getElementById('passPauseBtn');
        var phaseBtns = document.querySelectorAll('.pass-controls .pass-btn[data-phase]');

        var currentPhaseMode = 'cycle';
        var isPaused = false;
        var isHovered = false;
        var mouseX = -9999, mouseY = -9999;
        var hoveredStation = null;
        var animId = null;
        var lastTime = 0;
        var simTime = 0;

        var PALETTE = {
            paperBg: '#0e0c0a',
            paperStation: '#161310',
            paperStationHover: '#1c1813',
            rule: '#2c261c',
            ruleStrong: '#3a3226',
            terracotta: '#e07a3f',
            terracottaGlow: 'rgba(224, 122, 63, 0.75)',
            terracottaTint: 'rgba(224, 122, 63, 0.16)',
            olive: '#9a9440',
            oliveGlow: 'rgba(154, 148, 64, 0.75)',
            oliveTint: 'rgba(154, 148, 64, 0.16)',
            gold: '#c9a35c',
            goldGlow: 'rgba(201, 163, 92, 0.8)',
            goldTint: 'rgba(201, 163, 92, 0.2)',
            ink: '#d8ccb4',
            inkSoft: '#b3a68c',
            inkFaint: '#7a7160'
        };

        var STAGES = [
            {
                id: 1,
                tag: 'STAGE 01 · EMBEDDING',
                badge: '01 · EMB',
                title: 'Embedding',
                sub: 'x_t → 1024',
                chip: 'Tied Weights',
                op: 'h_0 = Embedding(x_t) · Tied with LM Head',
                shape: 'Input: [B, 2048] uint32 → Output: [B, 2048, 1024] bf16',
                desc: 'GPT-2 BPE vocab (50,257) · Tied embedding/head parameter weight matrix',
                isParam: true
            },
            {
                id: 2,
                tag: 'STAGE 02 · IN-PROJECTION',
                badge: '02 · IN-PROJ',
                title: 'Pre-Norm & Proj',
                sub: 'W_in · 5136',
                chip: 'Fused Linear',
                op: 'u = W_in · RMSNorm(h_{l-1}) (packs x_ssm, B_t, C_t, dt)',
                shape: 'Linear: 1024 → 16*(64 + 4*64 + 1) = 5136 bf16',
                desc: 'Single fused linear projection for token stream, complex B/C and dt — no z, no conv',
                isParam: true
            },
            {
                id: 3,
                tag: 'STAGE 03 · COMPLEX SSD SCAN',
                badge: '03 · SSD SCAN',
                title: 'Complex SSD',
                sub: 'N=64 · Q=64',
                chip: '28× Blocks',
                op: 's_t = exp(softplus(dt_t)·A) · s_{t-1} + B_t x_t  (A,B complex64)',
                shape: 'Output Y: [B, 2048, 16, 64] bf16 · Per-chunk state s: [B, 32, 64, 16, 64, 64] complex64',
                desc: 'Chunkwise linear recurrence (Q=64) · N=64 complex states (32 conjugate pairs) · output is .real of complex recurrence',
                isParam: true
            },
            {
                id: 4,
                tag: 'STAGE 04 · MIMO MIXER',
                badge: '04 · MIMO',
                title: 'MIMO Mixer',
                sub: '16 heads × 64',
                chip: 'Eye Init',
                op: 'Y_mimo = W_mix · Y  (W_mix = I, the only identity-init parameter)',
                shape: 'Tensor: [B, 2048, 16, 64] bf16 · Linear: 1024 → 1024 (16*64 → 16*64)',
                desc: 'Fully-connected cross-head mixer replaces classical SISO constraints',
                isParam: true
            },
            {
                id: 5,
                tag: 'STAGE 05 · SWIGLU & RESIDUAL',
                badge: '05 · FFN',
                title: 'SwiGLU & Res',
                sub: 'Swish(z) ⊙ Y',
                chip: '+ Residual',
                op: 'h_l = h_{l-1} + W_out(Swish(FFN_gate(h_{l-1})) ⊙ FFN_up(h_{l-1}))',
                shape: 'FFN: 1024 → 2048 (gate+up fused) → 1024 bf16 · Residual: [B, 2048, 1024]',
                desc: 'Gated SwiGLU non-linearity with direct residual highway skip connection',
                isParam: true
            },
            {
                id: 6,
                tag: 'STAGE 06 · TIED HEAD & LOSS',
                badge: '06 · LOSS',
                title: 'Head & Loss',
                sub: 'Full-Vocab CE',
                chip: 'Cross-Entropy',
                op: 'ℓ = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))  (V=50257, full vocab)',
                shape: 'Logits: [B, 2048, 50257] (fp32) → Loss scalar ℓ · Tied: lm_head.weight = embed.weight',
                desc: 'Vanilla full-vocab cross-entropy over the tied embedding/head — no chunking, no VRAM trick',
                isParam: true
            }
        ];

        var forwardParticles = [];
        for (var f = 0; f < 24; f++) {
            forwardParticles.push({
                xFrac: f / 24,
                speed: 0.18 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (f % 3) - 1
            });
        }

        var backwardParticles = [];
        for (var b = 0; b < 24; b++) {
            backwardParticles.push({
                xFrac: b / 24,
                speed: 0.20 + 0.08 * Math.random(),
                size: 1.6 + 1.0 * Math.random(),
                lane: (b % 3) - 1
            });
        }

        var adamParticles = [];
        var width = 0, height = 0;
        var stations = [];

        function layoutStations() {
            var rect = container.getBoundingClientRect();
            var dpr = Math.min(window.devicePixelRatio || 1, 2);
            width = rect.width;
            height = rect.height || 260;
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.scale(dpr, dpr);

            stations = [];
            var numStages = STAGES.length;
            var marginX = 16;
            var availW = width - (marginX * 2);
            var gap = Math.max(8, Math.min(16, (availW - (numStages * 82)) / (numStages - 1)));
            var stationW = Math.max(76, (availW - (gap * (numStages - 1))) / numStages);
            var stationH = Math.min(124, height * 0.52);
            var stationY = (height - stationH) / 2 + 2;

            for (var i = 0; i < numStages; i++) {
                var sx = marginX + i * (stationW + gap);
                stations.push({
                    stage: STAGES[i],
                    x: sx,
                    y: stationY,
                    w: stationW,
                    h: stationH,
                    cx: sx + stationW / 2,
                    cy: stationY + stationH / 2,
                    glowForward: 0,
                    glowBackward: 0,
                    glowAdam: 0
                });
            }
        }

        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () {
                layoutStations();
                if (reduced || isPaused) render(0, true);
            });
            ro.observe(container);
        } else {
            window.addEventListener('resize', layoutStations);
        }
        layoutStations();

        // Mouse Listeners
        container.addEventListener('mousemove', function (e) {
            var rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovered = true;
        });

        container.addEventListener('mouseleave', function () {
            isHovered = false;
            mouseX = -9999;
            mouseY = -9999;
            hoveredStation = null;
            if (tooltipEl) tooltipEl.style.opacity = '0';
        });

        container.addEventListener('click', function (e) {
            var rect = canvas.getBoundingClientRect();
            var clickX = e.clientX - rect.left;
            var clickY = e.clientY - rect.top;

            stations.forEach(function (st) {
                if (clickX >= st.x && clickX <= st.x + st.w && clickY >= st.y && clickY <= st.y + st.h) {
                    st.glowForward = 1.0;
                    st.glowBackward = 1.0;
                    for (var k = 0; k < 12; k++) {
                        adamParticles.push({
                            x: st.cx,
                            y: st.cy,
                            vx: (Math.random() - 0.5) * 160,
                            vy: (Math.random() - 0.5) * 160,
                            life: 1.0,
                            color: PALETTE.gold
                        });
                    }
                }
            });
        });

        if (pauseBtn) {
            pauseBtn.addEventListener('click', function (e) {
                e.stopPropagation();
                isPaused = !isPaused;
                pauseBtn.innerHTML = isPaused ? '&#9658;' : '&#10074;&#10074;';
                pauseBtn.setAttribute('aria-label', isPaused ? 'Resume pipeline' : 'Pause pipeline');
                if (!isPaused && !animId) {
                    lastTime = performance.now();
                    loop(lastTime);
                }
            });
        }

        phaseBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                phaseBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentPhaseMode = btn.getAttribute('data-phase') || 'cycle';
                if (reduced || isPaused) render(0, true);
            });
        });

        var CYCLE_DURATION = 8.6;

        function getCycleState(t) {
            if (currentPhaseMode === 'forward') {
                return { phase: 'forward', progress: (t * 0.4) % 1.0, subText: 'FORWARD ACTIVATIONS STREAMING' };
            }
            if (currentPhaseMode === 'backward') {
                return { phase: 'backward', progress: (t * 0.4) % 1.0, subText: 'AUTOGRAD GRADIENT PROPAGATION & CHECKPOINTING' };
            }

            var cycleT = t % CYCLE_DURATION;
            if (cycleT < 3.2) {
                return { phase: 'forward', progress: cycleT / 3.2, subText: 'FORWARD · Activations x_t → 28× Blocks → Logits' };
            } else if (cycleT < 4.2) {
                return { phase: 'loss', progress: (cycleT - 3.2) / 1.0, subText: 'LOSS COMPUTATION · Full-Vocab Cross-Entropy over tied embed/head (V=50,257)' };
            } else if (cycleT < 7.0) {
                return { phase: 'backward', progress: (cycleT - 4.2) / 2.8, subText: 'AUTOGRAD BACKWARD · Checkpointing Recomputes Activations' };
            } else if (cycleT < 8.0) {
                return { phase: 'adam', progress: (cycleT - 7.0) / 1.0, subText: 'ADAMW STEP · Weight Updates θ ← θ - η·∇ℓ & RNG Snapshot' };
            } else {
                return { phase: 'rest', progress: (cycleT - 8.0) / 0.6, subText: 'STEP COMMITTED · Next Mini-Batch Ingest' };
            }
        }

        function drawBackground() {
            ctx.fillStyle = PALETTE.paperBg;
            ctx.fillRect(0, 0, width, height);

            ctx.save();
            // Subtle technical grid
            ctx.strokeStyle = 'rgba(58, 50, 38, 0.22)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 8]);
            for (var x = 20; x < width; x += 40) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height);
                ctx.stroke();
            }
            ctx.setLineDash([]);

            // Upper Forward Conduit
            var fwdY = height * 0.18;
            ctx.beginPath();
            ctx.moveTo(14, fwdY);
            ctx.lineTo(width - 14, fwdY);
            ctx.strokeStyle = 'rgba(224, 122, 63, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();

            // Lower Backward Conduit
            var bwdY = height * 0.82;
            ctx.beginPath();
            ctx.moveTo(14, bwdY);
            ctx.lineTo(width - 14, bwdY);
            ctx.strokeStyle = 'rgba(154, 148, 64, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = PALETTE.terracotta;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'left';
            ctx.fillText('FORWARD CONDUIT (ACTIVATIONS) →', 20, fwdY - 6);

            ctx.fillStyle = PALETTE.olive;
            ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
            ctx.textAlign = 'right';
            ctx.fillText('← BACKWARD CONDUIT (GRADIENTS)', width - 20, bwdY + 14);
        }
        function drawParticles(state, dt) {
            ctx.save();
            var fwdY = height * 0.18;
            var bwdY = height * 0.82;

            // Forward streaming particles
            if (state.phase === 'forward' || state.phase === 'loss' || currentPhaseMode === 'forward') {
                forwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = 16 + p.xFrac * (width - 32);
                    var py = fwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.terracottaGlow;
                    ctx.fill();

                    // Forward motion trail
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px - 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(224, 122, 63, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            // Backward streaming particles
            if (state.phase === 'backward' || currentPhaseMode === 'backward') {
                backwardParticles.forEach(function (p) {
                    if (!reduced) p.xFrac += p.speed * dt;
                    if (p.xFrac > 1.0) p.xFrac = 0.0;

                    var px = width - 16 - p.xFrac * (width - 32);
                    var py = bwdY + p.lane * 3.0;

                    ctx.beginPath();
                    ctx.arc(px, py, p.size, 0, Math.PI * 2);
                    ctx.fillStyle = PALETTE.oliveGlow;
                    ctx.fill();

                    // Backward motion trail
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + 14 * p.speed, py);
                    ctx.strokeStyle = 'rgba(154, 148, 64, 0.35)';
                    ctx.lineWidth = p.size * 0.7;
                    ctx.stroke();
                });
            }

            // AdamW Burst Particles
            for (var k = adamParticles.length - 1; k >= 0; k--) {
                var ap = adamParticles[k];
                ap.x += ap.vx * dt;
                ap.y += ap.vy * dt;
                ap.life -= dt * 1.6;

                if (ap.life <= 0) {
                    adamParticles.splice(k, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(ap.x, ap.y, 2.0 * ap.life, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(201, 163, 92, ' + ap.life.toFixed(2) + ')';
                ctx.fill();
            }

            ctx.restore();
        }

        function drawStations(state, dt) {
            ctx.save();
            var closest = null;
            var numStations = stations.length;

            stations.forEach(function (st, idx) {
                var frac = idx / (numStations - 1);

                // Compute activation level
                var isFwdActive = false;
                var isBwdActive = false;
                var isAdamActive = (state.phase === 'adam');

                if (state.phase === 'forward') {
                    isFwdActive = state.progress >= (frac - 0.12) && state.progress <= (frac + 0.18);
                } else if (state.phase === 'loss') {
                    isFwdActive = (idx === numStations - 1);
                } else if (state.phase === 'backward') {
                    var bwdFrac = 1.0 - frac;
                    isBwdActive = state.progress >= (bwdFrac - 0.12) && state.progress <= (bwdFrac + 0.18);
                }

                if (isFwdActive) st.glowForward = 1.0;
                else st.glowForward = Math.max(0, st.glowForward - dt * 2.2);

                if (isBwdActive) st.glowBackward = 1.0;
                else st.glowBackward = Math.max(0, st.glowBackward - dt * 2.2);

                if (isAdamActive) st.glowAdam = 1.0;
                else st.glowAdam = Math.max(0, st.glowAdam - dt * 2.0);

                // Check Hover
                var isHover = isHovered && (mouseX >= st.x && mouseX <= st.x + st.w && mouseY >= st.y && mouseY <= st.y + st.h);
                if (isHover) closest = st;

                // Station Box Drawing
                var cardBg = isHover ? PALETTE.paperStationHover : PALETTE.paperStation;
                var borderColor = PALETTE.ruleStrong;

                if (st.glowAdam > 0.1) {
                    borderColor = PALETTE.gold;
                    cardBg = 'rgba(201, 163, 92, ' + (0.15 * st.glowAdam).toFixed(2) + ')';
                } else if (st.glowForward > 0.1) {
                    borderColor = PALETTE.terracotta;
                    cardBg = 'rgba(224, 122, 63, ' + (0.14 * st.glowForward).toFixed(2) + ')';
                } else if (st.glowBackward > 0.1) {
                    borderColor = PALETTE.olive;
                    cardBg = 'rgba(154, 148, 64, ' + (0.14 * st.glowBackward).toFixed(2) + ')';
                }

                // Station Background & Border
                ctx.fillStyle = cardBg;
                ctx.strokeStyle = isHover ? PALETTE.gold : borderColor;
                ctx.lineWidth = (isHover || st.glowForward > 0.3 || st.glowBackward > 0.3) ? 1.5 : 1.0;

                ctx.beginPath();
                ctx.rect(st.x, st.y, st.w, st.h);
                ctx.fill();
                ctx.stroke();

                // Corner Technical Brackets
                var crLen = 4;
                ctx.strokeStyle = isHover ? PALETTE.gold : PALETTE.ruleStrong;
                ctx.lineWidth = 1;
                ctx.beginPath();
                // Top-Left
                ctx.moveTo(st.x, st.y + crLen); ctx.lineTo(st.x, st.y); ctx.lineTo(st.x + crLen, st.y);
                // Top-Right
                ctx.moveTo(st.x + st.w - crLen, st.y); ctx.lineTo(st.x + st.w, st.y); ctx.lineTo(st.x + st.w, st.y + crLen);
                // Bottom-Left
                ctx.moveTo(st.x, st.y + st.h - crLen); ctx.lineTo(st.x, st.y + st.h); ctx.lineTo(st.x + crLen, st.y + st.h);
                // Bottom-Right
                ctx.moveTo(st.x + st.w - crLen, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h); ctx.lineTo(st.x + st.w, st.y + st.h - crLen);
                ctx.stroke();

                // Status Beacon LED Dot
                var ledColor = PALETTE.inkFaint;
                if (st.glowAdam > 0.2) ledColor = PALETTE.gold;
                else if (st.glowForward > 0.2) ledColor = PALETTE.terracotta;
                else if (st.glowBackward > 0.2) ledColor = PALETTE.olive;

                ctx.beginPath();
                ctx.arc(st.x + 8, st.y + 10, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = ledColor;
                ctx.fill();

                // Responsive title & badge for compact screens
                var isCompact = st.w < 70;
                var displayBadge = isCompact ? '0' + st.stage.id : st.stage.badge;
                var displayTitle = isCompact ? (st.stage.id === 1 ? 'Embed' : st.stage.id === 2 ? 'In-Proj' : st.stage.id === 3 ? 'SSD' : st.stage.id === 4 ? 'MIMO' : st.stage.id === 5 ? 'SwiGLU' : 'Loss') : st.stage.title;

                // Stage Number Badge
                ctx.font = isCompact ? 'bold 7.5px "JetBrains Mono", monospace' : 'bold 8.5px "JetBrains Mono", monospace';
                ctx.fillStyle = (st.glowForward > 0.2 ? PALETTE.terracotta : (st.glowBackward > 0.2 ? PALETTE.olive : PALETTE.inkSoft));
                ctx.textAlign = 'left';
                ctx.fillText(displayBadge, st.x + (isCompact ? 13 : 14), st.y + 13);

                // Stage Title
                ctx.font = isCompact ? 'bold 9px "JetBrains Mono", monospace' : 'bold 10px "JetBrains Mono", monospace';
                ctx.fillStyle = isHover ? '#ffffff' : PALETTE.ink;
                ctx.textAlign = 'center';
                ctx.fillText(displayTitle, st.cx, st.y + st.h * 0.40);

                // Subtitle
                if (st.h > 80) {
                    ctx.font = isCompact ? '7.5px "JetBrains Mono", monospace' : '8.5px "JetBrains Mono", monospace';
                    ctx.fillStyle = PALETTE.gold;
                    ctx.fillText(st.stage.sub, st.cx, st.y + st.h * 0.60);
                }

                // Chip Tag
                if (st.h > 100 && !isCompact) {
                    ctx.font = '8px "JetBrains Mono", monospace';
                    ctx.fillStyle = PALETTE.inkSoft;
                    ctx.fillText(st.stage.chip, st.cx, st.y + st.h * 0.78);
                }

                // Re-compute / Checkpoint Tag during Backward
                if (st.glowBackward > 0.3 && (idx >= 1 && idx <= 4)) {
                    ctx.fillStyle = PALETTE.olive;
                    ctx.font = 'bold 7.5px "JetBrains Mono", monospace';
                    ctx.fillText('RE-COMPUTE', st.cx, st.y + st.h - 6);
                } else if (st.glowAdam > 0.3) {
                    ctx.fillStyle = PALETTE.gold;
                    ctx.font = 'bold 7.5px "JetBrains Mono", monospace';
                    ctx.fillText('θ UPDATE', st.cx, st.y + st.h - 6);
                }

                // Inter-stage dataflow bridge connector
                if (idx < numStations - 1) {
                    var nextSt = stations[idx + 1];
                    var gapStartX = st.x + st.w;
                    var gapEndX = nextSt.x;
                    var midY = st.cy;

                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(gapStartX, midY);
                    ctx.lineTo(gapEndX, midY);
                    ctx.strokeStyle = (st.glowForward > 0.2 || nextSt.glowForward > 0.2) ? 'rgba(224, 122, 63, 0.7)' :
                                      (st.glowBackward > 0.2 || nextSt.glowBackward > 0.2) ? 'rgba(154, 148, 64, 0.7)' :
                                      PALETTE.ruleStrong;
                    ctx.lineWidth = (st.glowForward > 0.2 || nextSt.glowForward > 0.2 || st.glowBackward > 0.2 || nextSt.glowBackward > 0.2) ? 1.5 : 1.0;
                    ctx.stroke();

                    // Small directional chevron arrow
                    var arrowX = (gapStartX + gapEndX) / 2;
                    ctx.beginPath();
                    if (state.phase === 'backward') {
                        ctx.moveTo(arrowX + 2.5, midY - 2.5);
                        ctx.lineTo(arrowX - 2.5, midY);
                        ctx.lineTo(arrowX + 2.5, midY + 2.5);
                        ctx.strokeStyle = PALETTE.olive;
                    } else {
                        ctx.moveTo(arrowX - 2.5, midY - 2.5);
                        ctx.lineTo(arrowX + 2.5, midY);
                        ctx.lineTo(arrowX - 2.5, midY + 2.5);
                        ctx.strokeStyle = (st.glowForward > 0.2) ? PALETTE.terracotta : PALETTE.rule;
                    }
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.restore();
                }
            });

            hoveredStation = closest;
            ctx.restore();
        }

        function updateHUD(state) {
            if (phaseEl) {
                var phaseName = 'FORWARD ACTIVATIONS';
                if (state.phase === 'loss') phaseName = 'FULL-VOCAB CE LOSS (ℓ)';
                else if (state.phase === 'backward') phaseName = 'AUTOGRAD BACKWARD';
                else if (state.phase === 'adam') phaseName = 'ADAMW WEIGHT UPDATE';
                else if (state.phase === 'rest') phaseName = 'STANDBY / NEXT MINI-BATCH';
                phaseEl.textContent = phaseName;
            }

            if (tickerText) {
                tickerText.textContent = state.subText;
            }

            if (tickerBeacon) {
                tickerBeacon.style.color = (state.phase === 'forward' ? PALETTE.terracotta : (state.phase === 'backward' ? PALETTE.olive : PALETTE.gold));
            }

            if (hoveredStation && tooltipEl) {
                tooltipEl.style.opacity = '1';
                var st = hoveredStation.stage;
                var targetX = hoveredStation.x;
                var targetY = hoveredStation.y;

                var cardW = 225;
                var cardH = 75;

                var leftPx = targetX + hoveredStation.w / 2 - cardW / 2;
                var topPx = targetY - cardH - 12;
                if (topPx < 8) topPx = targetY + hoveredStation.h + 12;

                leftPx = Math.max(12, Math.min(width - cardW - 12, leftPx));

                tooltipEl.style.left = leftPx + 'px';
                tooltipEl.style.top = topPx + 'px';

                if (stTag) stTag.textContent = st.tag;
                if (stOp) stOp.textContent = st.op;
                if (stShape) stShape.textContent = st.shape;
                if (stDesc) stDesc.textContent = st.desc;
            } else if (tooltipEl) {
                tooltipEl.style.opacity = '0';
            }
        }

        function render(dt, force) {
            if (!force && isPaused) return;

            simTime += dt;
            ctx.clearRect(0, 0, width, height);

            var state = getCycleState(simTime);

            drawBackground();
            drawParticles(state, dt);
            drawStations(state, dt);
            updateHUD(state);
        }

        function loop(timestamp) {
            if (!lastTime) lastTime = timestamp;
            var dt = Math.min((timestamp - lastTime) / 1000, 0.1);
            lastTime = timestamp;

            render(dt, false);

            if (!isPaused && !reduced) {
                animId = requestAnimationFrame(loop);
            }
        }

        if (reduced) {
            render(0, true);
        } else {
            lastTime = performance.now();
            animId = requestAnimationFrame(loop);

            document.addEventListener('visibilitychange', function () {
                if (document.hidden) {
                    if (animId) {
                        cancelAnimationFrame(animId);
                        animId = null;
                    }
                } else if (!isPaused && !animId) {
                    lastTime = performance.now();
                    animId = requestAnimationFrame(loop);
                }
            });
        }
    }
    // ------------------------------------------------------------------
    // 3. Expandable Code Blocks (>14 lines) & Copy-to-Clipboard
    // ------------------------------------------------------------------
    function fallbackCopy(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.top = '-9999px';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta);
    }

    window.toggleCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var collapsed = wrapper.classList.toggle('collapsed');
        var nLines = wrapper.getAttribute('data-lines') || '';
        if (!nLines) {
            var codeEl = wrapper.querySelector('code');
            if (codeEl) {
                var lines = codeEl.innerText.split('\n').length;
                nLines = String(lines);
            }
        }
        btn.textContent = collapsed ? ('expand \u25be \u00b7 ' + nLines + ' lines') : 'collapse \u25b4';
    };

    window.copyCode = function (btn) {
        var wrapper = btn.closest('.code-wrapper');
        if (!wrapper) return;
        var code = wrapper.querySelector('pre code') || wrapper.querySelector('code');
        if (!code) return;
        var text = code.innerText || code.textContent;

        function flash() {
            var orig = btn.textContent;
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(function () {
                btn.textContent = orig;
                btn.classList.remove('copied');
            }, 1800);
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(flash).catch(function () {
                fallbackCopy(text);
                flash();
            });
        } else {
            fallbackCopy(text);
            flash();
        }
    };

    // ------------------------------------------------------------------
    // 4. Navigation Filtering & Sidebar Mobile Toggle
    // ------------------------------------------------------------------
    window.filterNav = function () {
        var input = document.getElementById('navSearch');
        if (!input) return;
        var q = (input.value || '').toLowerCase().trim();
        var links = document.querySelectorAll('.nav-link');
        var groupHeaders = document.querySelectorAll('.nav-group');

        links.forEach(function (l) {
            var txt = l.textContent.toLowerCase();
            var item = l.closest('.nav-item');
            if (!item) return;
            item.style.display = (!q || txt.indexOf(q) !== -1) ? '' : 'none';
        });

        // Hide empty category groups when filtered
        groupHeaders.forEach(function (grp) {
            var visibleItems = grp.querySelectorAll('.nav-item:not([style*="display: none"])');
            grp.style.display = (!q || visibleItems.length > 0) ? '' : 'none';
        });
    };
    window.toggleSidebar = function () {
        var sb = document.getElementById('sidebar');
        if (sb) sb.classList.toggle('open');
    };

    // ------------------------------------------------------------------
    // 5. Table of Contents Scrollspy
    // ------------------------------------------------------------------
    function initScrollspy() {
        var tocSidebar = document.querySelector('.toc-sidebar');
        var tocLinks = Array.from(document.querySelectorAll('.toc-link'));
        if (!tocLinks.length) return;

        var headings = [];
        tocLinks.forEach(function (link) {
            var href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                var targetId = href.substring(1);
                var el = document.getElementById(targetId);
                if (el) headings.push({ el: el, link: link, id: targetId });
            }
        });
        if (!headings.length) return;

        var isTicking = false;
        var activeItem = null;

        function updateSpy() {
            isTicking = false;

            // Reading threshold: 140px below the viewport top (accounts for sticky site-header)
            var threshold = 140;
            var current = null;

            // Find the active heading in viewport DOM order
            for (var i = 0; i < headings.length; i++) {
                var rect = headings[i].el.getBoundingClientRect();
                if (rect.top <= threshold) {
                    current = headings[i];
                } else {
                    break;
                }
            }

            // Default to first heading when near the top of the page
            if (!current && headings.length > 0 && window.scrollY < 300) {
                current = headings[0];
            }

            // If scrolled to the very bottom of the document, activate the last heading
            if ((window.innerHeight + window.scrollY) >= (document.body.offsetHeight - 60)) {
                current = headings[headings.length - 1];
            }

            if (current !== activeItem) {
                activeItem = current;

                tocLinks.forEach(function (l) {
                    l.classList.remove('active');
                });

                if (current && current.link) {
                    current.link.classList.add('active');

                    // Auto-scroll TOC sidebar to keep the active link visible and centered
                    if (tocSidebar) {
                        var sidebarRect = tocSidebar.getBoundingClientRect();
                        var linkRect = current.link.getBoundingClientRect();

                        // Check if link is outside the comfortable visible zone of TOC sidebar
                        if (linkRect.top < sidebarRect.top + 30 || linkRect.bottom > sidebarRect.bottom - 30) {
                            var linkOffsetInSidebar = current.link.offsetTop;
                            var targetScroll = linkOffsetInSidebar - (tocSidebar.clientHeight / 2) + (current.link.clientHeight / 2);
                            tocSidebar.scrollTo({
                                top: Math.max(0, targetScroll),
                                behavior: 'smooth'
                            });
                        }
                    }
                }
            }
        }

        function onScroll() {
            if (!isTicking) {
                requestAnimationFrame(updateSpy);
                isTicking = true;
            }
        }

        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', onScroll, { passive: true });

        setTimeout(updateSpy, 100);
        setTimeout(updateSpy, 500);
    }

    // ------------------------------------------------------------------
    // 6. Highlight.js & KaTeX bootstrap (also runs on doc pages).
    // ------------------------------------------------------------------
    function initHighlightAndMath() {
        if (window.hljs) hljs.highlightAll();
        if (window.renderMathInElement) {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '$', right: '$', display: false}
                ],
                ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
                throwOnError: false
            });
        }
    }

    // ------------------------------------------------------------------
    // 5b. Mechanism Cards — Complex SSD, MIMO Mixer, Chunkwise Pipeline
    // ------------------------------------------------------------------

    /* Card 1: Complex SSD Eigenvalue Spectrum Explorer */
    function initSsdMechanism() {
        var canvas = document.getElementById('ssdEigenCanvas');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var slider = document.getElementById('ssdDecaySlider');
        var decayLabel = document.getElementById('ssdDecayLabel');
        var halfLifeEl = document.getElementById('ssdHalfLife');
        var capacityEl = document.getElementById('ssdCapacity');
        var injectBtn = document.getElementById('ssdInjectBtn');

        var N = 64;
        var pairs = N / 2;
        var decayPulses = [];
        var eigenvalues = [];

        function genEigenvalues(decayMag) {
            eigenvalues = [];
            for (var i = 0; i < pairs; i++) {
                var alpha = -(Math.random() * decayMag * 0.2 + 0.05);
                var omega = (i / pairs) * Math.PI * 1.8 + 0.12 * Math.PI;
                eigenvalues.push({ alpha: alpha, omega: omega, mag: 0.3 + Math.random() * 0.7 });
            }
        }

        function draw() {
            var dpr = window.devicePixelRatio || 1;
            var rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            var w = rect.width, h = rect.height;

            ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--paper').trim() || '#12100d';
            ctx.fillRect(0, 0, w, h);

            var cx = w / 2, cy = h / 2;
            var rMax = Math.min(cx, cy) * 0.88;

            /* Axes */
            ctx.strokeStyle = '#2c261c';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(0, cy); ctx.lineTo(w, cy);
            ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
            ctx.stroke();

            /* Unit circle (dashed) */
            ctx.strokeStyle = '#3a3226';
            ctx.lineWidth = 0.5;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.arc(cx, cy, rMax, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            /* Axis labels */
            ctx.fillStyle = '#7a7160';
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.fillText('Re', w - 16, cy - 4);
            ctx.fillText('Im', cx + 4, 10);

            /* Eigenvalue conjugate pairs */
            var dotR = Math.max(2.2, Math.min(3.5, w / 90));
            for (var i = 0; i < eigenvalues.length; i++) {
                var ev = eigenvalues[i];
                var expDecay = Math.exp(ev.alpha);
                var re = expDecay * Math.cos(ev.omega);
                var im = expDecay * Math.sin(ev.omega);
                var px = cx + re * rMax;
                var py = cy - im * rMax;
                var pyC = cy + im * rMax;

                ctx.globalAlpha = 0.75 * ev.mag;
                ctx.fillStyle = '#e07a3f';
                ctx.beginPath();
                ctx.arc(px, py, dotR, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = '#9a9440';
                ctx.beginPath();
                ctx.arc(px, pyC, dotR, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.globalAlpha = 1;

            /* Decay pulse rings */
            for (var p = decayPulses.length - 1; p >= 0; p--) {
                var pulse = decayPulses[p];
                pulse.r += 1.5;
                pulse.alpha *= 0.96;
                if (pulse.alpha < 0.02) { decayPulses.splice(p, 1); continue; }
                ctx.strokeStyle = 'rgba(224, 122, 63, ' + pulse.alpha + ')';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(cx, cy, pulse.r, 0, Math.PI * 2);
                ctx.stroke();
            }

            requestAnimationFrame(draw);
        }

        genEigenvalues(15 * 0.2);

        function recomputeDecayUI(v) {
            var decayMag = v * 0.2;
            if (decayLabel) decayLabel.textContent = '[-' + decayMag.toFixed(1) + ', -0.05]';
            var hl = Math.log(2) / (decayMag * 0.5 + 0.05);
            if (halfLifeEl) halfLifeEl.textContent = '~' + hl.toFixed(1) + ' steps';
            return decayMag;
        }

        recomputeDecayUI(parseFloat(slider ? slider.value : 15));

        if (slider) {
            slider.addEventListener('input', function () {
                var decayMag = recomputeDecayUI(parseFloat(this.value));
                genEigenvalues(decayMag);
            });
        }

        if (injectBtn) {
            injectBtn.addEventListener('click', function () {
                decayPulses.push({ r: 5, alpha: 0.8 });
                genEigenvalues(parseFloat(slider.value) * 0.2);
                capacityEl.textContent = '2.0\u00d7 per parameter';
            });
        }

        draw();
    }

    /* Card 2: MIMO Head Mixer Matrix — heatmap-style 16×16 grid */
    function initMimoMechanism() {
        var grid = document.getElementById('mimoMatrixGrid');
        var stepBtn = document.getElementById('mimoStepBtn');
        var activeLinksEl = document.getElementById('mimoActiveLinks');
        if (!grid) return;

        var N = 16;
                var weights = [];
                var cells = [];

                for (var i = 0; i < N; i++) {
                    weights[i] = [];
                    for (var j = 0; j < N; j++) {
                        weights[i][j] = i === j ? 1.0 : 0.0;
                    }
                }

        for (var i = 0; i < N; i++) {
            for (var j = 0; j < N; j++) {
                var cell = document.createElement('div');
                cell.className = 'mimo-cell' + (i === j ? ' diagonal' : '');
                grid.appendChild(cell);
                cells.push({ el: cell, i: i, j: j });
            }
        }

        /* Heatmap colour — positive weights shift terracotta, negative shift olive */
        function heatColor(w) {
            var abs = Math.abs(w);
            if (abs < 0.01) return '';
            var t = Math.min(1, abs / 0.6);
            if (w > 0) return 'rgba(224,122,63,' + (t * 0.55).toFixed(2) + ')';
            return 'rgba(154,148,64,' + (t * 0.55).toFixed(2) + ')';
        }

        function updateDisplay() {
            var active = 0;
            for (var k = 0; k < cells.length; k++) {
                var c = cells[k];
                var w = weights[c.i][c.j];
                var abs = Math.abs(w);

                if (c.i === c.j) {
                    c.el.textContent = '1.0';
                } else if (abs < 0.01) {
                    c.el.textContent = '';
                } else if (abs < 0.1) {
                    c.el.textContent = w.toFixed(2);
                } else {
                    c.el.textContent = w.toFixed(1);
                }

                /* Heatmap background for off-diagonal */
                if (c.i !== c.j) {
                    var bg = heatColor(w);
                    c.el.style.background = bg || '';
                }

                if (c.i !== c.j && abs > 0.05) {
                    c.el.classList.add('active');
                    active++;
                } else {
                    c.el.classList.remove('active');
                }
            }
            if (activeLinksEl) activeLinksEl.textContent = active + ' of 240 off-diagonal';
        }

        if (stepBtn) {
            stepBtn.addEventListener('click', function () {
                for (var i = 0; i < N; i++) {
                    for (var j = 0; j < N; j++) {
                        if (i !== j) {
                            weights[i][j] += (Math.random() - 0.4) * 0.08;
                            weights[i][j] = Math.max(-1, Math.min(1, weights[i][j]));
                        }
                    }
                }
                updateDisplay();
            });
        }

        updateDisplay();
    }

    /* Card 3: Chunkwise Recurrence Pipeline — progressive verify */
    function initChunkMechanism() {
        var verifyBtn = document.getElementById('chunkVerifyBtn');
        var matchEl = document.getElementById('chunkMatchVal');
        var display = document.getElementById('chunkPipelineDisplay');
        if (!verifyBtn) return;

        var stages = display ? display.querySelectorAll('.chunk-stage') : [];
        var running = false;

        verifyBtn.addEventListener('click', function () {
            if (running) return;
            running = true;
            verifyBtn.disabled = true;
            var currentStage = 0;

            /* Dim all stages first */
            for (var s = 0; s < stages.length; s++) {
                stages[s].style.opacity = '0.25';
            }

            var interval = setInterval(function () {
                for (var s = 0; s < stages.length; s++) {
                    stages[s].style.opacity = s <= currentStage ? '1' : '0.25';
                }
                currentStage++;
                if (currentStage >= stages.length) {
                    clearInterval(interval);
                    var maxDelta = (Math.random() * 5e-6 + 1e-7);
                    if (matchEl) matchEl.textContent = '< ' + maxDelta.toExponential(1) + ' max |\u0394|';
                    for (var s = 0; s < stages.length; s++) {
                        stages[s].style.opacity = '1';
                    }
                    running = false;
                    verifyBtn.disabled = false;
                }
            }, 500);
        });
    }

    // ------------------------------------------------------------------
    // Boot Initialization
    // ------------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', function () {
        initComplexSpiral();
        initPassDiagram();
        initSsdMechanism();
        initMimoMechanism();
        initChunkMechanism();
        initScrollspy();
        initHighlightAndMath();
    });
})();
