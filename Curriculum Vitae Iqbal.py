from pathlib import Path
import zipfile

root = Path("/mnt/data/iqbal_portfolio")
(root / "components").mkdir(parents=True, exist_ok=True)
(root / "content" / "posts").mkdir(parents=True, exist_ok=True)
(root / "assets").mkdir(parents=True, exist_ok=True)

files = {
"app.py": r'''import streamlit as st
from pathlib import Path
import base64
import html

from components.ui import inject_css, navbar, section_title, card, timeline_item

st.set_page_config(
    page_title="Mohammad Iqbal Surya Ramadhan | Portfolio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()
navbar()

# ---------- HERO ----------
st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">INFORMATICS ENGINEERING • PORTFOLIO</div>
        <h1>Mohammad Iqbal<br><span>Surya Ramadhan</span></h1>
        <p class="hero-subtitle">
            Informatics Engineering student with an interest in programming,
            data analysis, technology, and interactive learning.
        </p>
        <div class="hero-actions">
            <a class="btn btn-primary" href="#journey">Explore My Journey</a>
            <a class="btn btn-ghost" href="#contact">Contact Me</a>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ---------- ABOUT ----------
section_title("01", "About Me", "A technical background shaped by real-world work experience.")

col1, col2 = st.columns([1.55, 1], gap="large")

with col1:
    st.markdown(
        """
        <div class="about-copy">
            <p>
            Mahasiswa Teknik Informatika yang proaktif, adaptif, dan komunikatif
            dengan pemahaman di bidang logika pemrograman, analisis data, dan
            sistem teknologi.
            </p>
            <p>
            Pengalaman di retail dan F&B membentuk kemampuan komunikasi publik,
            manajemen waktu, pelayanan pelanggan, teamwork, dan adaptasi di
            lingkungan kerja yang cepat.
            </p>
            <p>
            Saya juga memiliki ketertarikan untuk membagikan ilmu di bidang
            teknologi dan mengembangkan lingkungan belajar yang interaktif,
            kreatif, dan menyenangkan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        card(
            "Current Focus",
            [
                ("⌘", "Programming & Technology"),
                ("◈", "Data Analysis"),
                ("⚙", "Systems & Problem Solving"),
                ("✦", "Technology Education"),
            ],
        ),
        unsafe_allow_html=True,
    )

# ---------- EDUCATION ----------
section_title("02", "Education", "Academic journey.")

edu1, edu2 = st.columns(2, gap="large")
with edu1:
    st.markdown(
        card(
            "2021 — 2026",
            [("🎓", "Universitas Trunojoyo Madura"), ("", "Teknik Informatika")],
        ),
        unsafe_allow_html=True,
    )
with edu2:
    st.markdown(
        card(
            "2018 — 2021",
            [("🎓", "SMK Antartika 2 Sidoarjo"), ("", "Teknik Komputer dan Jaringan")],
        ),
        unsafe_allow_html=True,
    )

# ---------- JOURNEY ----------
section_title("03", "My Journey", "Experience across technology, retail, and F&B.")

st.markdown('<div id="journey"></div>', unsafe_allow_html=True)

timeline = [
    (
        "JAN 2026 — APR 2026",
        "Cook Helper",
        "Koat Coffee Sidoarjo",
        "Food preparation, portion control, menu knowledge, basic cooking, stock opname, inventory, teamwork, upselling/cross-selling, and maintaining hygiene standards.",
    ),
    (
        "AUG 2025 — DEC 2025",
        "Crew Outlet",
        "Pit-Stop Kopi Porong",
        "Handled barista, cashier, and kitchen responsibilities including coffee/non-coffee/milk-based beverages, customer service, preparation, outlet cleanliness, and upselling.",
    ),
    (
        "NOV 2024 — JUN 2025",
        "Operator & Kasir",
        "86 Printing — Retail ATK & Percetakan",
        "Customer service, sales transactions, printing-machine operation, Microsoft Office, simple design, binding, pamphlet/sticker/merchandise printing, product knowledge, and equipment maintenance.",
    ),
]

for date, role, company, description in timeline:
    st.markdown(
        timeline_item(date, role, company, description),
        unsafe_allow_html=True,
    )

# ---------- SKILLS ----------
section_title("04", "Skills", "Practical skills developed through study and work.")

skills = [
    ("Programming & Technology", "Logic programming, technology systems"),
    ("Data", "Data analysis"),
    ("Communication", "Public communication & customer service"),
    ("Teamwork", "Project team management"),
    ("Operations", "Stock opname, logistics & inventory"),
    ("F&B", "Food preparation, plating & presentation"),
    ("Barista", "Manual brew & espresso machine basics"),
    ("Work Style", "Discipline, accuracy, adaptability & time management"),
]

cols = st.columns(4, gap="medium")
for i, (title, desc) in enumerate(skills):
    with cols[i % 4]:
        st.markdown(
            f"""
            <div class="skill-card">
                <div class="skill-number">{i+1:02}</div>
                <h3>{html.escape(title)}</h3>
                <p>{html.escape(desc)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- PROJECTS ----------
section_title("05", "Projects", "A space for future technical work and experiments.")

p1, p2, p3 = st.columns(3, gap="medium")
projects = [
    ("01", "Portfolio Website", "Personal portfolio built with Streamlit.", "Python • Streamlit"),
    ("02", "Data Project", "A place to showcase data analysis and visualization work.", "Python • Data"),
    ("03", "Technology Project", "A place for future programming, system, or robotics projects.", "Technology • Learning"),
]
for col, (num, title, desc, tags) in zip([p1, p2, p3], projects):
    with col:
        st.markdown(
            f"""
            <div class="project-card">
                <span class="project-num">{num}</span>
                <h3>{html.escape(title)}</h3>
                <p>{html.escape(desc)}</p>
                <div class="tag">{html.escape(tags)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- BLOG ----------
section_title("06", "From The Blog", "Write about what you learn, build, and discover.")

posts_dir = Path("content/posts")
posts = sorted(posts_dir.glob("*.md")) if posts_dir.exists() else []

if posts:
    cols = st.columns(min(3, len(posts)), gap="medium")
    for i, post in enumerate(posts[:3]):
        text = post.read_text(encoding="utf-8")
        title = post.stem.replace("-", " ").title()
        preview = text.replace("#", "").strip().replace("\n", " ")[:130]
        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div class="blog-card">
                    <div class="blog-label">ARTICLE</div>
                    <h3>{html.escape(title)}</h3>
                    <p>{html.escape(preview)}...</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        """
        <div class="empty-blog">
            <span>✦</span>
            <h3>Your first article can live here.</h3>
            <p>Add Markdown files to <code>content/posts/</code> and this section will display them.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- CONTACT ----------
section_title("07", "Let's Connect", "Interested in collaborating, discussing technology, or sharing ideas?")

st.markdown('<div id="contact"></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")
contacts = [
    ("EMAIL", "moch.iqbalsury@gmail.com", "mailto:moch.iqbalsury@gmail.com"),
    ("PHONE", "+62 896 6575 9354", "tel:+6289665759354"),
    ("INSTAGRAM", "@Lilacnoe", "https://instagram.com/Lilacnoe"),
]
for col, (label, value, link) in zip([c1, c2, c3], contacts):
    with col:
        st.markdown(
            f"""
            <a class="contact-card" href="{link}" target="_blank">
                <small>{html.escape(label)}</small>
                <strong>{html.escape(value)}</strong>
            </a>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <footer>
        <div>MOHAMMAD IQBAL SURYA RAMADHAN</div>
        <span>Built with Streamlit • Personal Portfolio</span>
    </footer>
    """,
    unsafe_allow_html=True,
)
''',

"components/ui.py": r'''import streamlit as st
import html

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #0b0b10;
            --panel: #11111a;
            --panel-2: #151520;
            --text: #f4f1fa;
            --muted: #aaa5b7;
            --line: #272431;
            --accent: #c9a7ff;
            --accent-2: #9c72e8;
        }

        html { scroll-behavior: smooth; }

        .stApp {
            background:
                radial-gradient(circle at 80% 5%, rgba(139, 92, 246, .13), transparent 28%),
                radial-gradient(circle at 10% 35%, rgba(201, 167, 255, .06), transparent 25%),
                var(--bg);
            color: var(--text);
            font-family: 'Manrope', sans-serif;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { display: none; }

        .nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0 24px;
            border-bottom: 1px solid var(--line);
        }

        .brand {
            font-family: 'DM Mono', monospace;
            font-weight: 500;
            letter-spacing: .08em;
            color: var(--text);
        }

        .nav-links {
            display: flex;
            gap: 22px;
            font-size: .78rem;
            color: var(--muted);
            font-family: 'DM Mono', monospace;
        }

        .nav-links a {
            color: inherit;
            text-decoration: none;
            transition: .2s ease;
        }

        .nav-links a:hover { color: var(--accent); }

        .hero {
            min-height: 640px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 80px 0 100px;
            position: relative;
        }

        .hero:after {
            content: "✦";
            position: absolute;
            right: 5%;
            top: 25%;
            font-size: 11rem;
            color: rgba(201,167,255,.05);
            line-height: 1;
        }

        .hero-kicker, .blog-label, .skill-number, small {
            font-family: 'DM Mono', monospace;
            letter-spacing: .12em;
            color: var(--accent);
            font-size: .72rem;
        }

        .hero h1 {
            font-size: clamp(3.2rem, 8vw, 7.4rem);
            line-height: .94;
            letter-spacing: -.065em;
            margin: 20px 0 28px;
            font-weight: 800;
        }

        .hero h1 span { color: var(--accent); }

        .hero-subtitle {
            max-width: 680px;
            color: var(--muted);
            font-size: 1.08rem;
            line-height: 1.8;
        }

        .hero-actions { display: flex; gap: 12px; margin-top: 30px; }

        .btn {
            display: inline-block;
            text-decoration: none;
            padding: 13px 20px;
            border-radius: 8px;
            font-weight: 700;
            font-size: .82rem;
        }

        .btn-primary {
            background: var(--accent);
            color: #15111d;
        }

        .btn-ghost {
            border: 1px solid var(--line);
            color: var(--text);
        }

        .section-title {
            display: flex;
            align-items: baseline;
            gap: 18px;
            margin: 90px 0 35px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 18px;
        }

        .section-number {
            font-family: 'DM Mono', monospace;
            color: var(--accent);
            font-size: .75rem;
        }

        .section-title h2 {
            font-size: 2rem;
            margin: 0;
            letter-spacing: -.04em;
        }

        .section-sub {
            margin-left: auto;
            color: var(--muted);
            font-size: .82rem;
        }

        .about-copy p {
            color: #cbc7d2;
            line-height: 1.9;
            font-size: 1rem;
        }

        .info-card, .skill-card, .project-card, .blog-card {
            background: linear-gradient(145deg, rgba(255,255,255,.035), rgba(255,255,255,.015));
            border: 1px solid var(--line);
            border-radius: 14px;
        }

        .info-card { padding: 25px; }
        .info-title { color: var(--accent); font-family: 'DM Mono'; font-size: .75rem; margin-bottom: 18px; }
        .info-row { display: flex; gap: 12px; margin: 16px 0; color: #d5d0dd; }

        .timeline {
            border-left: 1px solid #3a3344;
            margin-left: 10px;
            padding-left: 28px;
        }

        .timeline-item {
            position: relative;
            padding: 5px 0 42px;
        }

        .timeline-item:before {
            content: "";
            position: absolute;
            left: -34px;
            top: 9px;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 0 5px rgba(201,167,255,.08);
        }

        .timeline-date {
            color: var(--accent);
            font: .72rem 'DM Mono';
            letter-spacing: .08em;
        }

        .timeline-role { font-size: 1.35rem; font-weight: 800; margin: 7px 0 3px; }
        .timeline-company { color: #d3cddc; font-weight: 600; }
        .timeline-desc { color: var(--muted); line-height: 1.7; max-width: 850px; margin-top: 10px; }

        .skill-card {
            padding: 20px;
            min-height: 150px;
            margin-bottom: 15px;
            transition: transform .2s ease, border-color .2s ease;
        }

        .skill-card:hover, .project-card:hover, .blog-card:hover {
            transform: translateY(-3px);
            border-color: #51435f;
        }

        .skill-card h3, .project-card h3, .blog-card h3 { margin: 14px 0 8px; font-size: 1rem; }
        .skill-card p, .project-card p, .blog-card p { color: var(--muted); line-height: 1.6; font-size: .82rem; }

        .project-card, .blog-card { padding: 25px; min-height: 220px; }
        .project-num { color: var(--accent); font: .75rem 'DM Mono'; }
        .tag {
            display: inline-block;
            margin-top: 22px;
            padding: 6px 9px;
            border: 1px solid var(--line);
            border-radius: 999px;
            color: #cbc4d6;
            font: .68rem 'DM Mono';
        }

        .empty-blog {
            padding: 45px;
            text-align: center;
            border: 1px dashed #383140;
            border-radius: 14px;
            color: var(--muted);
        }

        .empty-blog span { color: var(--accent); font-size: 2rem; }
        .empty-blog h3 { color: var(--text); }

        .contact-card {
            display: flex;
            flex-direction: column;
            gap: 9px;
            padding: 22px;
            border: 1px solid var(--line);
            border-radius: 12px;
            text-decoration: none;
            color: var(--text);
            background: rgba(255,255,255,.02);
        }

        .contact-card:hover { border-color: var(--accent-2); }
        .contact-card strong { font-size: .9rem; word-break: break-word; }

        footer {
            margin-top: 100px;
            padding-top: 24px;
            border-top: 1px solid var(--line);
            display: flex;
            justify-content: space-between;
            color: #777180;
            font: .7rem 'DM Mono';
        }

        @media (max-width: 700px) {
            .nav-links { display: none; }
            .hero { min-height: 550px; padding-top: 50px; }
            .section-title { display: block; }
            .section-sub { display: block; margin: 8px 0 0; }
            .hero-actions { flex-direction: column; max-width: 220px; }
            footer { flex-direction: column; gap: 10px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def navbar():
    st.markdown(
        """
        <div class="nav">
            <div class="brand">IQBAL.SR</div>
            <div class="nav-links">
                <a href="#about">ABOUT</a>
                <a href="#journey">JOURNEY</a>
                <a href="#projects">PROJECTS</a>
                <a href="#blog">BLOG</a>
                <a href="#contact">CONTACT</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_title(number, title, subtitle):
    anchor = title.lower().replace(" ", "-")
    st.markdown(
        f"""
        <div id="{anchor}" class="section-title">
            <span class="section-number">{html.escape(number)}</span>
            <h2>{html.escape(title)}</h2>
            <span class="section-sub">{html.escape(subtitle)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def card(title, rows):
    content = "".join(
        f'<div class="info-row"><span>{html.escape(icon)}</span><span>{html.escape(text)}</span></div>'
        for icon, text in rows
    )
    return f'<div class="info-card"><div class="info-title">{html.escape(title)}</div>{content}</div>'

def timeline_item(date, role, company, description):
    return f"""
    <div class="timeline">
        <div class="timeline-item">
            <div class="timeline-date">{html.escape(date)}</div>
            <div class="timeline-role">{html.escape(role)}</div>
            <div class="timeline-company">{html.escape(company)}</div>
            <div class="timeline-desc">{html.escape(description)}</div>
        </div>
    </div>
    """
''',

"components/__init__.py": "",
"content/posts/01-membangun-portfolio-streamlit.md": r'''# Membangun Portfolio dengan Streamlit

Ini adalah contoh artikel pertama. Ceritakan proses belajar, eksperimen, atau project yang sedang kamu kerjakan.

## Kenapa Streamlit?

Streamlit membuat proses membuat aplikasi berbasis Python menjadi cepat dan sederhana.

> Ganti artikel ini dengan tulisanmu sendiri.
''',

"requirements.txt": r'''streamlit>=1.40.0
''',

"README.md": r'''# Iqbal Personal Portfolio

Personal portfolio website berbasis Streamlit dengan gaya Modern Editorial + Dark Tech.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
