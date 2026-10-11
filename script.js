/* ═══════════════════════════════════════
   SolarGuard AI — JavaScript
   ═══════════════════════════════════════ */

// ── NAV Scroll Effect ──
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
});

// ── Hamburger Menu ──
const hamburger = document.getElementById('hamburger');
const navLinks  = document.getElementById('navLinks');
hamburger.addEventListener('click', () => {
  navLinks.classList.toggle('open');
});
document.querySelectorAll('.nav-links a').forEach(a => {
  a.addEventListener('click', () => navLinks.classList.remove('open'));
});

// ── Particle System ──
function createParticles() {
  const container = document.getElementById('heroParticles');
  const colors = ['#f5a623', '#ff6b35', '#06d6a0', '#ffd166', '#ffffff'];
  for (let i = 0; i < 45; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 4 + 1;
    p.style.cssText = `
      width: ${size}px;
      height: ${size}px;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      left: ${Math.random() * 100}%;
      top: ${Math.random() * 100}%;
      animation-duration: ${Math.random() * 12 + 8}s;
      animation-delay: ${Math.random() * -15}s;
      opacity: 0;
    `;
    container.appendChild(p);
  }
}
createParticles();

// ── Counter Animation ──
function animateCounter(el) {
  const target = parseInt(el.dataset.target, 10);
  const duration = 2000;
  const start = performance.now();
  const formatNum = n => n >= 10000 ? n.toLocaleString() : n.toString();

  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = formatNum(Math.round(eased * target));
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ── Bar Animation ──
function animateBars(entries) {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.querySelectorAll('.defect-fill, .sdg-fill').forEach(bar => {
      const pct = bar.style.getPropertyValue('--pct') || bar.style.width;
      bar.style.width = pct || bar.style.width;
    });
  });
}

// ── Intersection Observer ──
const io = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;

    const el = entry.target;

    // Counters
    if (el.classList.contains('stat-number')) {
      animateCounter(el);
      io.unobserve(el);
    }

    // Fade-in cards
    if (el.classList.contains('problem-card')   ||
        el.classList.contains('novelty-card')    ||
        el.classList.contains('sdg-card')        ||
        el.classList.contains('challenge-item')  ||
        el.classList.contains('ds-card-main')) {
      el.style.animationPlayState = 'running';
      io.unobserve(el);
    }

    // Bar fills
    if (el.classList.contains('defect-bar-section') ||
        el.classList.contains('sdg-grid')) {
      el.querySelectorAll('.defect-fill').forEach(bar => {
        setTimeout(() => { bar.style.width = bar.style.getPropertyValue('--pct'); }, 200);
      });
      el.querySelectorAll('.sdg-fill').forEach(bar => {
        setTimeout(() => { /* already set via inline style */ }, 200);
      });
      io.unobserve(el);
    }
  });
}, { threshold: 0.15 });

// Observe counters
document.querySelectorAll('.stat-number').forEach(el => io.observe(el));

// Observe animatable cards
document.querySelectorAll(
  '.problem-card, .novelty-card, .sdg-card, .challenge-item, .ds-card-main'
).forEach((el, i) => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(30px)';
  el.style.transition = `opacity .6s ease ${i * 0.08}s, transform .6s ease ${i * 0.08}s`;
  io.observe(el);
});

// Kick card animations immediately if they are already visible
function checkVisible(el) {
  const rect = el.getBoundingClientRect();
  return rect.top < window.innerHeight && rect.bottom > 0;
}

function revealCards() {
  document.querySelectorAll(
    '.problem-card, .novelty-card, .sdg-card, .challenge-item, .ds-card-main'
  ).forEach(el => {
    if (checkVisible(el) && el.style.opacity === '0') {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }
  });
}

// Trigger bar fills via intersection
const barSections = document.querySelectorAll('.defect-bar-section, .sdg-grid');
const barIO = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.querySelectorAll('.defect-fill').forEach(bar => {
      bar.style.width = bar.style.getPropertyValue('--pct');
    });
    barIO.unobserve(entry.target);
  });
}, { threshold: 0.2 });

barSections.forEach(el => barIO.observe(el));

// Scroll handler for card reveals
window.addEventListener('scroll', revealCards);
window.addEventListener('load',   revealCards);

// ── Smooth active nav link highlight ──
const sections = document.querySelectorAll('section[id], header[id]');
const navAnchors = document.querySelectorAll('.nav-links a');

const navIO = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navAnchors.forEach(a => {
        a.style.color = '';
        if (a.getAttribute('href') === `#${entry.target.id}`) {
          a.style.color = 'var(--gold)';
        }
      });
    }
  });
}, { threshold: 0.4 });

sections.forEach(s => navIO.observe(s));

// ── Panel Grid: Defect pulse random ──
const cells = document.querySelectorAll('.panel-cell');
cells.forEach(cell => {
  if (!cell.classList.contains('healthy')) return;
  setInterval(() => {
    const r = Math.random();
    if (r < 0.05) {
      cell.style.boxShadow = 'inset 0 0 16px rgba(245,166,35,0.5)';
      setTimeout(() => { cell.style.boxShadow = ''; }, 800);
    }
  }, 1500);
});

// ── Explore button bounce ──
const exploreBtn = document.getElementById('exploreBtn');
let bounced = false;
setTimeout(() => {
  if (!bounced) {
    exploreBtn.style.animation = 'none';
  }
}, 5000);
