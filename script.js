// ─── State ───────────────────────────────────────────────────────────────────
const TOTAL_PAGES = 9;
let currentPage = 0;
let isTransitioning = false;

// Per-page Three.js contexts
const canvases = {}; // pageIndex → { scene, camera, renderer, mesh, animId }

// ─── Page IDs ─────────────────────────────────────────────────────────────────
const PAGE_IDS = [
    'page-hero',
    'page-motivation-1',
    'page-motivation-2',
    'page-motivation-3',
    'page-quickstart',
    'page-architecture',
    'page-features',
    'page-applications',
    'page-cta'
];

// ─── Particle themes per page ─────────────────────────────────────────────────
const PARTICLE_THEMES = [
    // 0 Hero: sphere — clean, iconic
    { type: 'sphere', count: 10000, color: 0x3d3929, size: 0.015, rotY: 0.003 },
    // 1-3 Motivation: brain/cloud — organic, flowing
    { type: 'cloud', count: 8000, color: 0x3d3929, size: 0.012, rotY: 0.002 },
    { type: 'cloud', count: 8000, color: 0x3d3929, size: 0.012, rotY: 0.002 },
    { type: 'cloud', count: 8000, color: 0x3d3929, size: 0.012, rotY: 0.002 },
    // 4 Quick-start: no canvas (text-focused page)
    null,
    // 5 Architecture: lattice — structured, modular
    { type: 'lattice', count: 6000, color: 0x3d3929, size: 0.014, rotY: 0.002 },
    // 6 Features: no canvas (cards page)
    null,
    // 7 Applications: torus — continuous, looping
    { type: 'torus', count: 7000, color: 0x3d3929, size: 0.013, rotY: 0.003 },
    // 8 CTA: no canvas
    null
];

// Canvas IDs for pages that have them
const CANVAS_IDS = [
    'hero-canvas',
    'motivation-canvas-1',
    'motivation-canvas-2',
    'motivation-canvas-3',
    null,
    'architecture-canvas',
    null,
    null,
    null
];

const CONTAINER_IDS = [
    'hero-canvas-container',
    'motivation-canvas-container-1',
    'motivation-canvas-container-2',
    'motivation-canvas-container-3',
    null,
    'architecture-canvas-container',
    null,
    null,
    null
];

// ─── Geometry builders ────────────────────────────────────────────────────────
function buildGeometry(type, count) {
    const pos = [];

    switch (type) {
        case 'sphere':
            for (let i = 0; i < count; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                const r = 1 + Math.random() * 0.15;
                pos.push(
                    r * Math.sin(phi) * Math.cos(theta),
                    r * Math.sin(phi) * Math.sin(theta),
                    r * Math.cos(phi)
                );
            }
            break;

        case 'cloud': {
            // Organic blob — sum of several offset spheres
            const centers = [
                [0, 0, 0], [0.6, 0.3, 0], [-0.5, 0.4, 0.2],
                [0.2, -0.5, 0.3], [-0.3, -0.3, -0.4]
            ];
            for (let i = 0; i < count; i++) {
                const c = centers[Math.floor(Math.random() * centers.length)];
                const r = 0.5 + Math.random() * 0.5;
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                pos.push(
                    c[0] + r * Math.sin(phi) * Math.cos(theta),
                    c[1] + r * Math.sin(phi) * Math.sin(theta),
                    c[2] + r * Math.cos(phi)
                );
            }
            break;
        }

        case 'lattice': {
            // 3D grid with slight jitter — modular feel
            const n = Math.ceil(Math.cbrt(count));
            const step = 2.2 / n;
            let added = 0;
            for (let x = 0; x < n && added < count; x++) {
                for (let y = 0; y < n && added < count; y++) {
                    for (let z = 0; z < n && added < count; z++) {
                        pos.push(
                            -1.1 + x * step + (Math.random() - 0.5) * step * 0.3,
                            -1.1 + y * step + (Math.random() - 0.5) * step * 0.3,
                            -1.1 + z * step + (Math.random() - 0.5) * step * 0.3
                        );
                        added++;
                    }
                }
            }
            break;
        }

        case 'torus':
            for (let i = 0; i < count; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.random() * Math.PI * 2;
                const R = 1, r = 0.38;
                pos.push(
                    (R + r * Math.cos(phi)) * Math.cos(theta),
                    (R + r * Math.cos(phi)) * Math.sin(theta),
                    r * Math.sin(phi)
                );
            }
            break;

        default:
            for (let i = 0; i < count; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                pos.push(Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi));
            }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    return geo;
}

// ─── Canvas init ──────────────────────────────────────────────────────────────
function initCanvas(pageIndex) {
    const theme = PARTICLE_THEMES[pageIndex];
    const canvasId = CANVAS_IDS[pageIndex];
    const containerId = CONTAINER_IDS[pageIndex];
    if (!theme || !canvasId || !containerId) return;

    const container = document.getElementById(containerId);
    const canvas = document.getElementById(canvasId);
    if (!container || !canvas) return;

    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    camera.position.z = 5;

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(w, h);
    renderer.setClearColor(0x000000, 0);

    const geo = buildGeometry(theme.type, theme.count);
    const mat = new THREE.PointsMaterial({ color: theme.color, size: theme.size, sizeAttenuation: true });
    const mesh = new THREE.Points(geo, mat);
    scene.add(mesh);

    const pageEl = document.getElementById(PAGE_IDS[pageIndex]);

    const onResize = () => {
        const rw = container.clientWidth || window.innerWidth;
        const rh = container.clientHeight || window.innerHeight;
        camera.aspect = rw / rh;
        camera.updateProjectionMatrix();
        renderer.setSize(rw, rh);
    };
    window.addEventListener('resize', onResize);

    // Store entry before starting animate loop so animate() can reference it safely
    const entry = { scene, camera, renderer, mesh, animId: null, onResize };
    canvases[pageIndex] = entry;

    function animate() {
        entry.animId = requestAnimationFrame(animate);
        if (!pageEl.classList.contains('active')) return;
        mesh.rotation.y += theme.rotY;
        renderer.render(scene, camera);
    }
    animate();
}

function safeInitCanvas(pageIndex) {
    try {
        initCanvas(pageIndex);
    } catch (error) {
        console.warn('Canvas unavailable for page', pageIndex, error);
    }
}

// ─── Page navigation ──────────────────────────────────────────────────────────
function navigateToPage(targetIndex) {
    if (isTransitioning || targetIndex === currentPage) return;
    if (targetIndex < 0 || targetIndex >= TOTAL_PAGES) return;

    isTransitioning = true;

    const fromEl = document.getElementById(PAGE_IDS[currentPage]);
    const toEl = document.getElementById(PAGE_IDS[targetIndex]);

    fromEl.classList.add('transitioning-out');

    setTimeout(() => {
        fromEl.classList.remove('active', 'transitioning-out');
        toEl.classList.add('active');
        currentPage = targetIndex;

        updatePageDots();
        updateBackToTop();

        // Lazy-init canvas for target page
        if (!canvases[targetIndex] && PARTICLE_THEMES[targetIndex]) {
            safeInitCanvas(targetIndex);
        }

        setTimeout(() => { isTransitioning = false; }, 100);
    }, 800);
}

// ─── Page dots ────────────────────────────────────────────────────────────────
function updatePageDots() {
    document.querySelectorAll('.page-dot').forEach((dot, i) => {
        dot.classList.toggle('active', i === currentPage);
        dot.setAttribute('aria-current', i === currentPage ? 'page' : 'false');
    });
}

function initPageDots() {
    document.querySelectorAll('.page-dot').forEach(dot => {
        dot.addEventListener('click', () => {
            const idx = parseInt(dot.dataset.page, 10);
            navigateToPage(idx);
        });
    });
}

// ─── Scroll / keyboard / touch navigation ────────────────────────────────────
function initNavigation() {
    document.body.style.overflow = 'hidden';

    let lastScrollTime = 0;
    const SCROLL_DELAY = 1200;

    window.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (isTransitioning) return;
        const now = Date.now();
        if (now - lastScrollTime < SCROLL_DELAY) return;
        lastScrollTime = now;
        navigateToPage(currentPage + (e.deltaY > 0 ? 1 : -1));
    }, { passive: false });

    window.addEventListener('keydown', (e) => {
        if (isTransitioning) return;
        if (e.key === 'ArrowDown' || e.key === 'PageDown') {
            e.preventDefault();
            navigateToPage(currentPage + 1);
        } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
            e.preventDefault();
            navigateToPage(currentPage - 1);
        }
    });

    // Touch swipe
    let touchStartY = 0;
    window.addEventListener('touchstart', (e) => { touchStartY = e.touches[0].clientY; }, { passive: true });
    window.addEventListener('touchend', (e) => {
        if (isTransitioning) return;
        const dy = touchStartY - e.changedTouches[0].clientY;
        if (Math.abs(dy) < 40) return;
        const now = Date.now();
        if (now - lastScrollTime < SCROLL_DELAY) return;
        lastScrollTime = now;
        navigateToPage(currentPage + (dy > 0 ? 1 : -1));
    }, { passive: true });
}

// ─── Back to top ──────────────────────────────────────────────────────────────
function updateBackToTop() {
    const btn = document.getElementById('back-to-top');
    if (!btn) return;
    btn.classList.toggle('hidden', currentPage === 0);
}

function initBackToTop() {
    const btn = document.getElementById('back-to-top');
    if (!btn) return;
    btn.addEventListener('click', () => navigateToPage(0));
}

// ─── Feature detail cards ────────────────────────────────────────────────────
let lastFeatureTrigger = null;

function openDetail(card, trigger) {
    const overlay = document.getElementById('feature-detail-overlay');
    const detailCard = overlay ? overlay.querySelector('.feature-detail-card') : null;
    const image = document.getElementById('feature-detail-image');
    const video = document.getElementById('feature-detail-video');
    const caption = document.getElementById('feature-detail-caption');
    if (!card || !overlay) return;

    lastFeatureTrigger = trigger || null;
    const hasImage = Boolean(card.detailImage);
    const hasVideo = Boolean(card.detailVideo);
    const hasTable = Boolean(card.detailTable);
    const media = hasImage ? card.detailImage : (hasVideo ? card.detailVideo : null);
    if (detailCard) {
        detailCard.classList.toggle('feature-detail-card--with-image', hasImage);
        detailCard.classList.toggle('feature-detail-card--with-video', hasVideo);
        detailCard.classList.toggle('feature-detail-card--with-table', hasTable);
    }
    set('feature-detail-num', card.num || '');
    set('feature-detail-title', card.title);
    set('feature-detail-body', card.body);
    set('feature-detail-list', (card.details || []).map(detail => `<li>${detail}</li>`).join(''));
    if (image) {
        image.hidden = !hasImage;
        image.src = hasImage ? card.detailImage.src : '';
        image.alt = hasImage ? card.detailImage.alt : '';
    }
    if (video) {
        video.hidden = !hasVideo;
        if (hasVideo) {
            const v = card.detailVideo;
            video.src = v.src;
            video.setAttribute('aria-label', v.alt || '');
            video.muted = true;
            video.loop = true;
            video.playsInline = true;
            video.play().catch(() => {});
        } else {
            video.pause();
            video.removeAttribute('src');
            video.load();
        }
    }
    if (caption) {
        const showCaption = Boolean(media && media.notCherryPicked);
        caption.hidden = !showCaption;
        caption.textContent = showCaption ? 'Not cherry-picked — actual model output.' : '';
    }
    const link = document.getElementById('feature-detail-link');
    if (link) {
        const hasLink = Boolean(card.detailLink);
        link.hidden = !hasLink;
        if (hasLink) {
            link.href = card.detailLink.href;
            link.innerHTML = card.detailLink.label;
        } else {
            link.removeAttribute('href');
            link.innerHTML = '';
        }
    }
    const tableWrap = document.getElementById('feature-detail-table');
    if (tableWrap) {
        tableWrap.hidden = !hasTable;
        if (hasTable) {
            const t = card.detailTable;
            const head = `<tr>${t.headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
            const body = t.rows.map((row, i) => {
                const cls = i === t.highlightRow ? ' class="is-highlight"' : '';
                return `<tr${cls}>${row.map(c => `<td>${c}</td>`).join('')}</tr>`;
            }).join('');
            tableWrap.innerHTML = `${t.caption ? `<p class="feature-detail-table-caption">${t.caption}</p>` : ''}<table class="feature-detail-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
        } else {
            tableWrap.innerHTML = '';
        }
    }

    overlay.classList.add('active');
    overlay.setAttribute('aria-hidden', 'false');
    const closeBtn = overlay.querySelector('[data-feature-close]');
    if (closeBtn) closeBtn.focus({ preventScroll: true });
}

function closeFeatureDetail() {
    const overlay = document.getElementById('feature-detail-overlay');
    if (!overlay || !overlay.classList.contains('active')) return;

    overlay.classList.remove('active');
    overlay.setAttribute('aria-hidden', 'true');
    const video = document.getElementById('feature-detail-video');
    if (video) {
        video.pause();
        video.removeAttribute('src');
        video.load();
    }
    if (lastFeatureTrigger) lastFeatureTrigger.focus({ preventScroll: true });
    lastFeatureTrigger = null;
}

function initFeatureDetails() {
    const featuresGrid = document.getElementById('features-track');
    const appsGrid = document.getElementById('app-cards');
    const overlay = document.getElementById('feature-detail-overlay');
    if (!overlay) return;

    const handleOpen = (cardEl, dataset, list) => {
        const idx = Number(cardEl.dataset[dataset]);
        const card = list[idx];
        if (!card) return;
        openDetail(card, cardEl);
    };

    if (featuresGrid) {
        featuresGrid.addEventListener('click', (e) => {
            if (e.target.closest('a')) return;
            const card = e.target.closest('.feature-card');
            if (!card) return;
            handleOpen(card, 'featureIndex', CONTENT.features.cards);
        });
        featuresGrid.addEventListener('keydown', (e) => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            if (e.target.closest('a')) return;
            const card = e.target.closest('.feature-card');
            if (!card) return;
            e.preventDefault();
            handleOpen(card, 'featureIndex', CONTENT.features.cards);
        });
    }

    if (appsGrid) {
        appsGrid.addEventListener('click', (e) => {
            if (e.target.closest('a')) return;
            const card = e.target.closest('.app-card');
            if (!card) return;
            handleOpen(card, 'appIndex', CONTENT.applications.cards);
        });
        appsGrid.addEventListener('keydown', (e) => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            if (e.target.closest('a')) return;
            const card = e.target.closest('.app-card');
            if (!card) return;
            e.preventDefault();
            handleOpen(card, 'appIndex', CONTENT.applications.cards);
        });
    }

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay || e.target.closest('[data-feature-close]')) {
            closeFeatureDetail();
        }
    });

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeFeatureDetail();
    });
}

// ─── Code copy button ─────────────────────────────────────────────────────────
function bindCopyButton(buttonId, codeId) {
    const btn = document.getElementById(buttonId);
    const code = document.getElementById(codeId);
    if (!btn || !code) return;

    btn.addEventListener('click', async () => {
        const text = code.textContent
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>');
        try {
            await navigator.clipboard.writeText(text);
            btn.textContent = 'copied!';
            setTimeout(() => { btn.textContent = 'copy'; }, 2000);
        } catch {
            btn.textContent = 'error';
            setTimeout(() => { btn.textContent = 'copy'; }, 2000);
        }
    });
}

function initCopyButton() {
    bindCopyButton('copy-btn', 'quickstart-code');
    bindCopyButton('citation-copy-btn', 'cta-citation');
}

function initCitationScroll() {
    const scroller = document.querySelector('.citation-code');
    const slider = document.getElementById('citation-scroll');
    if (!scroller || !slider) return;

    const maxScroll = () => Math.max(0, scroller.scrollWidth - scroller.clientWidth);
    const syncSlider = () => {
        const max = maxScroll();
        slider.value = max > 0 ? String((scroller.scrollLeft / max) * 100) : '0';
        slider.disabled = max === 0;
    };

    slider.addEventListener('input', () => {
        scroller.scrollLeft = maxScroll() * (Number(slider.value) / 100);
    });
    scroller.addEventListener('scroll', syncSlider);
    window.addEventListener('resize', syncSlider);
    requestAnimationFrame(syncSlider);
}

// ─── Audio controls ───────────────────────────────────────────────────────────
function initAudioControls() {
    const toggle = document.getElementById('audio-toggle');
    const audio = document.getElementById('site-audio');
    const label = toggle ? toggle.parentElement.querySelector('span') : null;
    if (!toggle || !audio) return;

    const TARGET_VOL = 0.35;
    const FADE_MS = 600;
    audio.volume = TARGET_VOL;
    let playing = false;
    let fadeRaf = null;

    function fadeTo(target, done) {
        if (fadeRaf) cancelAnimationFrame(fadeRaf);
        const start = audio.volume, t0 = performance.now();
        (function step(now) {
            const t = Math.min(1, (now - t0) / FADE_MS);
            audio.volume = start + (target - start) * t;
            if (t < 1) fadeRaf = requestAnimationFrame(step);
            else { fadeRaf = null; if (done) done(); }
        })(t0);
    }

    audio.addEventListener('error', () => {
        toggle.disabled = true;
        if (label) label.textContent = 'AUDIO UNAVAILABLE';
    });

    toggle.addEventListener('click', async () => {
        if (!playing) {
            audio.volume = TARGET_VOL;
            try {
                await audio.play();
                playing = true;
                toggle.classList.add('playing');
                toggle.setAttribute('aria-pressed', 'true');
                toggle.setAttribute('aria-label', 'Pause audio');
            } catch { if (label) label.textContent = 'TAP AGAIN'; }
        } else {
            fadeTo(0, () => { audio.pause(); audio.volume = TARGET_VOL; });
            playing = false;
            toggle.classList.remove('playing');
            toggle.setAttribute('aria-pressed', 'false');
            toggle.setAttribute('aria-label', 'Play audio');
        }
    });

    document.addEventListener('visibilitychange', () => {
        if (document.hidden && playing) audio.pause();
        else if (!document.hidden && playing) audio.play().catch(() => {});
    });
}

// ─── Easing ───────────────────────────────────────────────────────────────────
function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// ─── Content injection ────────────────────────────────────────────────────────
function set(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function highlightQuickstartCode(code) {
    return code.split('\n').map(line => {
        const commentIndex = line.indexOf('#');
        const codePart = commentIndex >= 0 ? line.slice(0, commentIndex) : line;
        const commentPart = commentIndex >= 0 ? line.slice(commentIndex) : '';
        let html = escapeHtml(codePart);

        html = html.replace(/^(\s*)(python)(\s+)(src\/main\.py)/, '$1<span class="code-token code-token--command">$2</span>$3<span class="code-token code-token--path">$4</span>');
        html = html.replace(/([A-Za-z0-9_.-]+)(=)([A-Za-z0-9_./+-]+)/g, '<span class="code-token code-token--key">$1</span>$2<span class="code-token code-token--value">$3</span>');
        html = html.replace(/(\\)(\s*)$/, '<span class="code-token code-token--slash">$1</span>$2');

        if (commentPart) {
            html += `<span class="code-token code-token--comment">${escapeHtml(commentPart)}</span>`;
        }

        return html;
    }).join('\n');
}

function highlightCitationCode(code) {
    return escapeHtml(code)
        .replace(/^(@[a-z]+)(\{)([^,\n]+)(,?)/i, '<span class="citation-token citation-token--type">$1</span>$2<span class="citation-token citation-token--id">$3</span>$4')
        .replace(/^(\s*)([a-z]+)(=)/gmi, '$1<span class="citation-token citation-token--key">$2</span>$3')
        .replace(/(\\url)(\{[^}]+\})/g, '<span class="citation-token citation-token--command">$1</span><span class="citation-token citation-token--url">$2</span>');
}

function injectContent() {
    const c = CONTENT;

    // Hero
    set('hero-title',    c.hero.title);
    set('hero-subtitle', c.hero.subtitle);
    document.querySelectorAll('[data-link="arxiv"]').forEach(a => { a.href = c.hero.links.arxiv; });
    document.querySelectorAll('[data-link="github"]').forEach(a => { a.href = c.hero.links.github; });
    document.querySelectorAll('[data-link="models"]').forEach(a => { a.href = c.hero.links.models; });
    document.querySelectorAll('[data-link="twitter"]').forEach(a => { a.href = c.hero.links.twitter; });

    // Motivation (3 pages: intro pair, science shift, NanoWM close)
    const motivationParagraphs = Array.isArray(c.motivation.body) ? c.motivation.body : [];
    const motivationPages = [
        motivationParagraphs.slice(0, 2),
        motivationParagraphs.slice(2, 3),
        motivationParagraphs.slice(3, 4),
    ];
    for (let i = 0; i < motivationPages.length; i++) {
        set(`motivation-label-${i + 1}`, c.motivation.label);
        set(`motivation-title-${i + 1}`, c.motivation.title);
        const pageParagraphs = motivationPages[i];
        set(`motivation-body-${i + 1}`, pageParagraphs.map(p => `<p class="page-body">${p}</p>`).join(''));

        const quoteEl = document.getElementById(`motivation-quote-${i + 1}`);
        if (quoteEl) {
            const quote = Array.isArray(c.motivation.quotes)
                ? c.motivation.quotes[i]
                : c.motivation.quote;
            if (quote && String(quote).trim().length > 0) {
                quoteEl.innerHTML = quote;
                quoteEl.style.display = '';
            } else {
                quoteEl.innerHTML = '';
                quoteEl.style.display = 'none';
            }
        }
    }

    // Quick-start
    set('quickstart-label', c.quickstart.label);
    set('quickstart-title', c.quickstart.title);
    set('quickstart-intro', c.quickstart.intro);
    const codeEl = document.getElementById('quickstart-code');
    if (codeEl) codeEl.innerHTML = highlightQuickstartCode(c.quickstart.code);
    set('quickstart-note',  c.quickstart.note);

    // Architecture
    set('arch-label', c.architecture.label);
    set('arch-title', c.architecture.title);
    set('arch-body',  c.architecture.body);
    const archTreeItems = c.architecture.tree.map((item, index, arr) => `
        <li class="arch-tree__item${index === arr.length - 1 ? ' arch-tree__item--last' : ''}">
            <span class="arch-tree__branch" aria-hidden="true"></span>
            <span class="arch-tree__name">${escapeHtml(item.name)}</span>
            <span class="arch-tree__desc">${escapeHtml(item.desc)}</span>
        </li>
    `).join('');
    set('arch-modules', `
        <div class="arch-tree" aria-label="src directory tree">
            <div class="arch-tree__root">src/</div>
            <ol class="arch-tree__list">${archTreeItems}</ol>
        </div>
        <div class="arch-buttons" aria-label="Core source modules">
            ${c.architecture.modules.map(m => `
            <div class="arch-module">
                <span class="arch-module__num">${m.num}</span>
                <div>
                    <h3 class="arch-module__title">${m.title}</h3>
                    <p class="arch-module__desc">${m.desc}</p>
                </div>
            </div>
            `).join('')}
        </div>`);

    // Features
    set('features-label', c.features.label);
    set('features-title', c.features.title);
    set('features-hint', c.features.interactionHint || '');
    set('features-track', c.features.cards.map((card, index) => `
        <div class="feature-card" role="button" tabindex="0" data-feature-index="${index}" aria-label="Open details for ${card.title}">
            <span class="feature-card__num">${card.num}</span>
            <h3 class="feature-card__title">${card.title}</h3>
            <p class="feature-card__body">${card.body}</p>
        </div>`).join(''));
    // Applications
    set('applications-label', c.applications.label);
    set('applications-title', c.applications.title);
    set('applications-hint', c.applications.interactionHint || '');
    set('app-cards', c.applications.cards.map((card, index) => `
        <div class="app-card" role="button" tabindex="0" data-app-index="${index}" aria-label="Open details for ${card.title}">
            <span class="app-card__icon" aria-hidden="true">${card.icon}</span>
            <h3 class="app-card__title">${card.title}</h3>
            <p class="app-card__body">${card.body}</p>
        </div>`).join(''));

    // CTA
    set('cta-label',    c.cta.label);
    set('cta-title',    c.cta.title);
    set('cta-body',     c.cta.body);
    const citationEl = document.getElementById('cta-citation');
    if (citationEl) citationEl.innerHTML = highlightCitationCode(c.cta.citation);
    set('cta-copyright', c.cta.copyright);
    document.querySelectorAll('[data-link="cta-arxiv"]').forEach(a => { a.href = c.cta.links.arxiv; });
    document.querySelectorAll('[data-link="cta-github"]').forEach(a => { a.href = c.cta.links.github; });
    document.querySelectorAll('[data-link="cta-models"]').forEach(a => { a.href = c.cta.links.models; });
}

// ─── Boot ─────────────────────────────────────────────────────────────────────
const MOBILE_QUERY = '(max-width: 900px)';
function isMobileLayout() {
    return window.matchMedia(MOBILE_QUERY).matches;
}

document.addEventListener('DOMContentLoaded', () => {
    injectContent();

    if (isMobileLayout()) {
        // Mobile: stack all pages, native scrolling, no snap nav
        document.body.classList.add('is-mobile');
        document.querySelectorAll('.page-section').forEach(s => s.classList.add('active'));
        // Only init hero canvas to keep mobile GPU light
        safeInitCanvas(0);
    } else {
        safeInitCanvas(0);
        initNavigation();
        initPageDots();
        initBackToTop();
        updatePageDots();
        updateBackToTop();
    }

    initFeatureDetails();
    initCopyButton();
    initCitationScroll();
    initAudioControls();
});
