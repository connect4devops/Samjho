"""Generates realistic sample image assets for Scribble2Structure.

Generates 6 distinct test/sample images covering all required transformation modes:
1. Handwritten machine learning notes (gradient descent, loss formula, regularization, margin uncertainty)
2. Operating system process-state diagram (New -> Ready -> Running -> Terminated, Waiting)
3. Software architecture sketch (Client -> API Gateway -> Services -> DB, Cache, Queue)
4. Student productivity brainstorming map (Central theme with branches and categories)
5. Meeting notes containing tasks, owners, and deadlines
6. Database ER diagram (Entities, attributes, PK/FK, relationships)
"""

import os
from PIL import Image, ImageDraw, ImageFont


def _draw_grid(draw: ImageDraw.ImageDraw, w: int, h: int, step: int = 30, color: str = "#f0f2f5"):
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=color, width=1)


def _draw_lined_paper(draw: ImageDraw.ImageDraw, w: int, h: int, line_step: int = 34):
    # Margin line
    draw.line([(90, 0), (90, h)], fill="#fca5a5", width=2)
    # Horizontal ruled lines
    for y in range(70, h, line_step):
        draw.line([(0, y), (w, y)], fill="#e2e8f0", width=1)


def create_ml_notes(out_path: str):
    """Sample 1: Handwritten machine learning notes."""
    w, h = 1100, 780
    im = Image.new("RGB", (w, h), "#fcfbf7")
    draw = ImageDraw.Draw(im)
    _draw_lined_paper(draw, w, h, line_step=38)

    # Title
    draw.text((120, 25), "Machine Learning: Gradient Descent & Regularization", fill="#1e293b")
    draw.line([(120, 55), (750, 55)], fill="#64748b", width=2)

    # Content
    y = 80
    lines = [
        ("Core Concept: Optimizing Parameter Weights w and Bias b", "#0f172a", False),
        ("Cost Function: J(w,b) = (1 / 2m) * sum( (y_hat^(i) - y^(i))^2 )", "#1e3a8a", True),
        ("Gradient Descent Update Rule:", "#0f172a", False),
        ("  w := w - alpha * (dJ/dw)", "#1e3a8a", True),
        ("  b := b - alpha * (dJ/db)", "#1e3a8a", True),
        ("Hyperparameter Notes:", "#0f172a", False),
        ("  - alpha = 0.01 (learning rate - too large => diverges, too small => slow)", "#334155", False),
        ("  - batch size = 32 (mini-batch SGD chosen over full batch)", "#334155", False),
        ("Regularization (L2 / Ridge) to Prevent Overfitting:", "#b91c1c", False),
        ("  J_reg(w,b) = J(w,b) + (lambda / 2m) * sum(w_j^2)", "#1e3a8a", True),
        ("  Weight decay formula: w := w * (1 - alpha*lambda/m) - alpha * dJ/dw", "#334155", False),
        ("Bias vs Variance:", "#0f172a", False),
        ("  - High bias => Underfitting (solution: bigger model, more features)", "#475569", False),
        ("  - High variance => Overfitting (solution: more data, L2, dropout)", "#475569", False),
    ]

    for text, color, highlight in lines:
        if highlight:
            draw.rectangle([(115, y - 2), (820, y + 26)], outline="#93c5fd", width=1, fill="#eff6ff")
        draw.text((120, y), text, fill=color)
        y += 42

    # Uncertain / smudged margin text
    draw.text((860, 180), "[smudge: lr schedule?]", fill="#94a3b8")
    draw.text((860, 210), "decay = 0.95/epoch?", fill="#94a3b8")
    draw.text((860, 360), "[illegible: lambda=?]", fill="#dc2626")
    draw.text((860, 390), "check cross-validation", fill="#94a3b8")

    im.save(out_path, format="PNG")


def create_os_process(out_path: str):
    """Sample 2: Operating system process-state diagram."""
    w, h = 1100, 750
    im = Image.new("RGB", (w, h), "#f8fafc")
    draw = ImageDraw.Draw(im)
    _draw_grid(draw, w, h, 35, "#e2e8f0")

    # Title
    draw.text((50, 30), "OS Process State Transition Model", fill="#0f172a")
    draw.line([(50, 60), (450, 60)], fill="#3b82f6", width=3)

    # State boxes
    boxes = {
        "New": (80, 240, 200, 310, "#dbeafe", "#1e40af"),
        "Ready": (360, 240, 480, 310, "#dcfce7", "#166534"),
        "Running": (660, 240, 780, 310, "#fef3c7", "#92400e"),
        "Waiting": (510, 470, 650, 540, "#fee2e2", "#991b1b"),
        "Terminated": (930, 240, 1060, 310, "#f1f5f9", "#475569"),
    }

    for name, (x1, y1, x2, y2, bg, border) in boxes.items():
        draw.rectangle([(x1, y1), (x2, y2)], fill=bg, outline=border, width=2)
        # Center text roughly
        cx = (x1 + x2) // 2 - (len(name) * 4)
        cy = (y1 + y2) // 2 - 8
        draw.text((cx, cy), name, fill=border)

    # Arrows and Labels
    # New -> Ready
    draw.line([(200, 275), (360, 275)], fill="#1e293b", width=2)
    draw.polygon([(360, 275), (350, 270), (350, 280)], fill="#1e293b")
    draw.text((235, 250), "admitted", fill="#475569")

    # Ready -> Running (dispatch)
    draw.line([(480, 260), (660, 260)], fill="#1e293b", width=2)
    draw.polygon([(660, 260), (650, 255), (650, 265)], fill="#1e293b")
    draw.text((515, 238), "scheduler dispatch", fill="#166534")

    # Running -> Ready (interrupt)
    draw.line([(660, 290), (480, 290)], fill="#1e293b", width=2)
    draw.polygon([(480, 290), (490, 285), (490, 295)], fill="#1e293b")
    draw.text((535, 295), "interrupt", fill="#b45309")

    # Running -> Waiting (I/O or event wait)
    draw.line([(720, 310), (720, 505), (650, 505)], fill="#1e293b", width=2)
    draw.polygon([(650, 505), (660, 500), (660, 510)], fill="#1e293b")
    draw.text((725, 410), "I/O or event wait", fill="#991b1b")

    # Waiting -> Ready (I/O or event completion)
    draw.line([(510, 505), (420, 505), (420, 310)], fill="#1e293b", width=2)
    draw.polygon([(420, 310), (415, 320), (425, 320)], fill="#1e293b")
    draw.text((310, 485), "I/O completion", fill="#15803d")

    # Running -> Terminated (exit)
    draw.line([(780, 275), (930, 275)], fill="#1e293b", width=2)
    draw.polygon([(930, 275), (920, 270), (920, 280)], fill="#1e293b")
    draw.text((830, 250), "exit", fill="#475569")

    # Annotations / notes on whiteboard
    draw.text((80, 600), "PCB (Process Control Block): PID, Program Counter, CPU registers, memory limits", fill="#334155")
    draw.text((80, 630), "Context Switch overhead: save state of old proc -> load state of new proc", fill="#334155")
    draw.text((800, 600), "[ambiguous: zombie vs orphan?]", fill="#94a3b8")

    im.save(out_path, format="PNG")


def create_architecture(out_path: str):
    """Sample 3: Software architecture sketch."""
    w, h = 1100, 750
    im = Image.new("RGB", (w, h), "#0f172a")
    draw = ImageDraw.Draw(im)
    _draw_grid(draw, w, h, 35, "#1e293b")

    # Title
    draw.text((50, 30), "Cloud E-Commerce Platform Architecture", fill="#f8fafc")
    draw.line([(50, 60), (480, 60)], fill="#38bdf8", width=3)

    # Client layer
    draw.rectangle([(60, 220), (200, 340)], fill="#1e293b", outline="#38bdf8", width=2)
    draw.text((80, 250), "Client Tier", fill="#38bdf8")
    draw.text((75, 280), "Web SPA (React)", fill="#94a3b8")
    draw.text((75, 305), "Mobile App (iOS/And)", fill="#94a3b8")

    # API Gateway
    draw.rectangle([(280, 180), (440, 380)], fill="#1e293b", outline="#a855f7", width=2)
    draw.text((300, 210), "API Gateway", fill="#c084fc")
    draw.text((300, 250), "- Reverse Proxy", fill="#e2e8f0")
    draw.text((300, 280), "- JWT Auth / RBAC", fill="#e2e8f0")
    draw.text((300, 310), "- Rate Limiter (Redis)", fill="#e2e8f0")
    draw.text((300, 340), "- TLS Termination", fill="#e2e8f0")

    # Services
    draw.rectangle([(520, 120), (700, 220)], fill="#1e293b", outline="#34d399", width=2)
    draw.text((540, 145), "Product Catalog", fill="#34d399")
    draw.text((540, 175), "Port: 8081 / gRPC", fill="#94a3b8")

    draw.rectangle([(520, 260), (700, 360)], fill="#1e293b", outline="#34d399", width=2)
    draw.text((540, 285), "Order Service", fill="#34d399")
    draw.text((540, 315), "Port: 8082 / REST", fill="#94a3b8")

    draw.rectangle([(520, 400), (700, 500)], fill="#1e293b", outline="#34d399", width=2)
    draw.text((540, 425), "Payment Service", fill="#34d399")
    draw.text((540, 455), "Port: 8083 / REST", fill="#94a3b8")

    # Data layer
    draw.rectangle([(800, 120), (980, 220)], fill="#1e293b", outline="#fbbf24", width=2)
    draw.text((820, 145), "PostgreSQL DB", fill="#fbbf24")
    draw.text((820, 175), "Primary + Replica", fill="#94a3b8")

    draw.rectangle([(800, 260), (980, 360)], fill="#1e293b", outline="#f43f5e", width=2)
    draw.text((820, 285), "Redis Cluster", fill="#fb7185")
    draw.text((820, 315), "Cache + PubSub", fill="#94a3b8")

    draw.rectangle([(800, 400), (980, 500)], fill="#1e293b", outline="#f59e0b", width=2)
    draw.text((820, 425), "Kafka Event Bus", fill="#fbbf24")
    draw.text((820, 455), "Topics: orders, pay", fill="#94a3b8")

    # Arrows
    # Client -> Gateway
    draw.line([(200, 280), (280, 280)], fill="#e2e8f0", width=2)
    draw.polygon([(280, 280), (270, 275), (270, 285)], fill="#e2e8f0")
    draw.text((215, 260), "HTTPS", fill="#38bdf8")

    # Gateway -> Services
    for sy in [170, 310, 450]:
        draw.line([(440, 280), (480, 280), (480, sy), (520, sy)], fill="#c084fc", width=2)
        draw.polygon([(520, sy), (510, sy - 5), (510, sy + 5)], fill="#c084fc")

    # Services -> Data
    draw.line([(700, 170), (800, 170)], fill="#34d399", width=2)
    draw.polygon([(800, 170), (790, 165), (790, 175)], fill="#34d399")

    draw.line([(700, 310), (800, 310)], fill="#34d399", width=2)
    draw.polygon([(800, 310), (790, 305), (790, 315)], fill="#34d399")

    draw.line([(700, 450), (800, 450)], fill="#34d399", width=2)
    draw.polygon([(800, 450), (790, 445), (790, 455)], fill="#34d399")

    # Bottom notes
    draw.text((60, 620), "Architecture Principles: Microservices, Event-Driven Async Order Processing", fill="#94a3b8")
    draw.text((60, 650), "Uncertain: Stripe webhook secret rotation cadence not documented [?]", fill="#f87171")

    im.save(out_path, format="PNG")


def create_mindmap(out_path: str):
    """Sample 4: Student productivity brainstorming map."""
    w, h = 1100, 750
    im = Image.new("RGB", (w, h), "#fefce8")
    draw = ImageDraw.Draw(im)
    _draw_grid(draw, w, h, 35, "#fef08a")

    # Title
    draw.text((50, 30), "Brainstorm: Student Productivity System", fill="#854d0e")
    draw.line([(50, 60), (480, 60)], fill="#ca8a04", width=3)

    # Center Oval
    draw.ellipse([(430, 310), (670, 430)], fill="#fde047", outline="#854d0e", width=3)
    draw.text((475, 350), "Student", fill="#713f12")
    draw.text((455, 375), "Productivity", fill="#713f12")

    # Branches and Nodes
    branches = [
        # Study Habits (top-left)
        (200, 160, "Study Habits", "#bbf7d0", "#166534", [
            "Pomodoro (25/5 min cycles)",
            "Active Recall testing",
            "Spaced Repetition (Anki decks)"
        ], (470, 330)),
        # Task Management (top-right)
        (800, 160, "Task Management", "#bfdbfe", "#1e40af", [
            "Daily 3 Most Important Tasks",
            "Notion Kanban Board",
            "Eisenhower Matrix (Urgent vs Imp)"
        ], (630, 330)),
        # Health & Energy (bottom-left)
        (200, 520, "Health & Energy", "#fecaca", "#991b1b", [
            "7.5-8 hours consistent sleep",
            "Morning 20-min cardio/walk",
            "2L daily hydration goal"
        ], (470, 410)),
        # Tech & Tools (bottom-right)
        (800, 520, "Tools & Workspace", "#e9d5ff", "#6b21a8", [
            "Obsidian for linked knowledge",
            "Google Calendar time blocking",
            "Noise cancelling headphones"
        ], (630, 410)),
    ]

    for bx, by, cat, bg, border, items, anchor in branches:
        # Curve connection to center
        draw.line([anchor, (bx + 80, by + 20)], fill=border, width=2)

        draw.rectangle([(bx, by), (bx + 200, by + 45)], fill=bg, outline=border, width=2)
        draw.text((bx + 15, by + 12), cat, fill=border)

        iy = by + 60
        for item in items:
            draw.text((bx, iy), "• " + item, fill="#1f2937")
            iy += 26

    # Open questions / priority notes
    draw.text((60, 680), "Open Question: Digital vs Paper planner for quick captures? [unresolved]", fill="#b45309")
    draw.text((650, 680), "Suggested Priority: Fix sleep schedule before adding new tools", fill="#15803d")

    im.save(out_path, format="PNG")


def create_meeting_notes(out_path: str):
    """Sample 5: Meeting notes containing tasks and deadlines."""
    w, h = 1100, 780
    im = Image.new("RGB", (w, h), "#ffffff")
    draw = ImageDraw.Draw(im)
    _draw_lined_paper(draw, w, h, line_step=38)

    # Title
    draw.text((120, 25), "Sprint 42 Sync & Action Items — 10/24", fill="#0f172a")
    draw.line([(120, 55), (600, 55)], fill="#2563eb", width=2)

    y = 80
    sections = [
        ("Attendees: Sarah (PM), David (Backend), Elena (Frontend), Marcus (DevOps)", "#475569"),
        ("", "#000000"),
        ("KEY DECISIONS:", "#1e40af"),
        ("  1. Migrate authentication service to OAuth 2.0 / OIDC standard", "#1e293b"),
        ("  2. Deprecate legacy REST v1 API by end of Q4", "#1e293b"),
        ("  3. Freeze feature additions 2 weeks before Black Friday load test", "#1e293b"),
        ("", "#000000"),
        ("CONFIRMED ACTION ITEMS:", "#166534"),
        ("  [ ] Update OpenAPI 3.0 specs and publish docs  | Owner: David   | Due: 10/28", "#0f172a"),
        ("  [ ] Implement JWT refresh token rotation logic  | Owner: Elena   | Due: 11/02", "#0f172a"),
        ("  [ ] Provision staging Redis cluster for cache   | Owner: Marcus  | Due: Friday", "#0f172a"),
        ("  [ ] Sign off on revised security compliance docs| Owner: Sarah   | Due: 11/05", "#0f172a"),
        ("", "#000000"),
        ("UNRESOLVED QUESTIONS / UNASSIGNED:", "#b91c1c"),
        ("  [ ] Fix memory leak in ingest background worker | Owner: ??? [unclear] | Due: ASAP", "#b91c1c"),
        ("  ? Will v1 deprecation break mobile app build v2.4 users? (David checking)", "#94a3b8"),
        ("  ? Budget approval for Datadog APM upgrade pending finance review", "#94a3b8"),
    ]

    for line, color in sections:
        if line:
            draw.text((120, y), line, fill=color)
        y += 38

    im.save(out_path, format="PNG")


def create_database_er(out_path: str):
    """Sample 6: Database ER diagram."""
    w, h = 1100, 750
    im = Image.new("RGB", (w, h), "#f8fafc")
    draw = ImageDraw.Draw(im)
    _draw_grid(draw, w, h, 35, "#e2e8f0")

    # Title
    draw.text((50, 30), "E-Commerce Relational Schema (ER Diagram)", fill="#0f172a")
    draw.line([(50, 60), (520, 60)], fill="#6366f1", width=3)

    # Entities
    entities = [
        ("CUSTOMER", 80, 160, 280, 380, [
            ("customer_id", "INT", "PK"),
            ("full_name", "VARCHAR(100)", ""),
            ("email", "VARCHAR(150)", "UNIQUE"),
            ("created_at", "TIMESTAMP", ""),
        ]),
        ("ORDER", 440, 160, 660, 420, [
            ("order_id", "INT", "PK"),
            ("customer_id", "INT", "FK"),
            ("order_date", "TIMESTAMP", ""),
            ("total_amount", "DECIMAL(10,2)", ""),
            ("status", "VARCHAR(20)", ""),
        ]),
        ("ORDER_ITEM", 820, 160, 1040, 420, [
            ("item_id", "INT", "PK"),
            ("order_id", "INT", "FK"),
            ("product_id", "INT", "FK"),
            ("quantity", "INT", ""),
            ("unit_price", "DECIMAL(10,2)", ""),
        ]),
        ("PRODUCT", 820, 490, 1040, 710, [
            ("product_id", "INT", "PK"),
            ("sku", "VARCHAR(50)", "UNIQUE"),
            ("title", "VARCHAR(200)", ""),
            ("price", "DECIMAL(10,2)", ""),
            ("stock_qty", "INT", ""),
        ]),
    ]

    for name, x1, y1, x2, y2, attrs in entities:
        # Header
        draw.rectangle([(x1, y1), (x2, y1 + 35)], fill="#4f46e5", outline="#4338ca", width=2)
        draw.text((x1 + 15, y1 + 8), name, fill="#ffffff")
        # Body
        draw.rectangle([(x1, y1 + 35), (x2, y2)], fill="#ffffff", outline="#4338ca", width=2)
        ay = y1 + 45
        for col, col_type, key in attrs:
            prefix = f"[{key}] " if key else "     "
            draw.text((x1 + 10, ay), f"{prefix}{col}", fill="#1e293b")
            draw.text((x2 - 110, ay), col_type, fill="#64748b")
            ay += 32

    # Relationship connectors
    # CUSTOMER (1) -- (N) ORDER
    draw.line([(280, 250), (440, 250)], fill="#4338ca", width=2)
    draw.text((310, 225), "places (1:N)", fill="#4338ca")
    # Crow's foot notation hint
    draw.polygon([(440, 250), (425, 243), (425, 257)], fill="#4338ca")

    # ORDER (1) -- (N) ORDER_ITEM
    draw.line([(660, 250), (820, 250)], fill="#4338ca", width=2)
    draw.text((690, 225), "contains (1:N)", fill="#4338ca")
    draw.polygon([(820, 250), (805, 243), (805, 257)], fill="#4338ca")

    # PRODUCT (1) -- (N) ORDER_ITEM
    draw.line([(930, 490), (930, 420)], fill="#4338ca", width=2)
    draw.text((940, 450), "ordered in (1:N)", fill="#4338ca")
    draw.polygon([(930, 420), (923, 435), (937, 435)], fill="#4338ca")

    # Ambiguity note
    draw.text((80, 650), "Notes: Soft deletes vs hard deletes? Ambiguous refund relationship not shown [?]", fill="#dc2626")

    im.save(out_path, format="PNG")


def main():
    os.makedirs("samples", exist_ok=True)
    print("Generating sample 1: ML Notes...")
    create_ml_notes("samples/01_ml_notes.png")
    print("Generating sample 2: OS Process...")
    create_os_process("samples/02_os_process.png")
    print("Generating sample 3: Architecture...")
    create_architecture("samples/03_software_architecture.png")
    print("Generating sample 4: Mindmap...")
    create_mindmap("samples/04_student_productivity.png")
    print("Generating sample 5: Meeting Notes...")
    create_meeting_notes("samples/05_meeting_notes.png")
    print("Generating sample 6: Database ER...")
    create_database_er("samples/06_database_er.png")
    print("All 6 samples successfully created in samples/")


if __name__ == "__main__":
    main()
