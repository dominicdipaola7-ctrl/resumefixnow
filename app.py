import os
import json
import uuid
import hashlib
import time
from datetime import datetime, date, timedelta
from functools import wraps

import stripe
import PyPDF2
from openai import OpenAI
from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for, abort
)
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from fpdf import FPDF
import io
import base64

load_dotenv()

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///resumeai.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_FILE_SIZE_MB", 10)) * 1024 * 1024

db = SQLAlchemy(app)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

FREE_DAILY_LIMIT = int(os.getenv("FREE_DAILY_LIMIT", 1))
STRIPE_PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID", "")
APP_URL = os.getenv("APP_URL", "http://localhost:5000")

# ── Models ─────────────────────────────────────────────────────────────────────
class FreeUsage(db.Model):
    __tablename__ = "free_usage"
    id = db.Column(db.Integer, primary_key=True)
    ip_hash = db.Column(db.String(64), nullable=False, index=True)
    usage_date = db.Column(db.Date, nullable=False, default=date.today)
    count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("ip_hash", "usage_date", name="uq_ip_date"),
    )


class ProSubscription(db.Model):
    __tablename__ = "pro_subscriptions"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False, unique=True, index=True)
    stripe_customer_id = db.Column(db.String(100), unique=True)
    stripe_subscription_id = db.Column(db.String(100), unique=True)
    status = db.Column(db.String(30), default="active")  # active, canceled, past_due
    pro_token = db.Column(db.String(64), unique=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)


class ResumeJob(db.Model):
    __tablename__ = "resume_jobs"
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    ip_hash = db.Column(db.String(64))
    pro_token = db.Column(db.String(64), nullable=True)
    is_pro = db.Column(db.Boolean, default=False)
    match_score = db.Column(db.Integer, nullable=True)
    keywords_found = db.Column(db.Text, nullable=True)   # JSON list
    keywords_missing = db.Column(db.Text, nullable=True) # JSON list
    rewritten_resume = db.Column(db.Text, nullable=True)
    cover_letter = db.Column(db.Text, nullable=True)
    ats_tips = db.Column(db.Text, nullable=True)         # JSON list
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_ip_hash():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    ip = ip.split(",")[0].strip()
    return hashlib.sha256(ip.encode()).hexdigest()


def check_free_limit():
    """Returns (allowed: bool, remaining: int)"""
    ip_hash = get_ip_hash()
    today = date.today()
    record = FreeUsage.query.filter_by(ip_hash=ip_hash, usage_date=today).first()
    if not record:
        return True, FREE_DAILY_LIMIT
    remaining = max(0, FREE_DAILY_LIMIT - record.count)
    return remaining > 0, remaining


def consume_free_usage():
    ip_hash = get_ip_hash()
    today = date.today()
    record = FreeUsage.query.filter_by(ip_hash=ip_hash, usage_date=today).first()
    if record:
        record.count += 1
    else:
        record = FreeUsage(ip_hash=ip_hash, usage_date=today, count=1)
        db.session.add(record)
    db.session.commit()


def validate_pro_token(token):
    if not token:
        return None
    sub = ProSubscription.query.filter_by(pro_token=token, status="active").first()
    if sub and (sub.expires_at is None or sub.expires_at > datetime.utcnow()):
        return sub
    return None


def extract_text_from_pdf(file_bytes):
    """Extract text content from PDF bytes."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        raise ValueError(f"Could not read PDF: {str(e)}")


def ai_analyze_resume(resume_text, job_description, is_pro=False):
    """Call OpenAI to analyze, rewrite resume, score match, and optionally generate cover letter."""

    system_prompt = """You are an expert resume writer and ATS optimization specialist with 15+ years of experience helping candidates land interviews at top companies. You analyze resumes against job descriptions and provide comprehensive, actionable rewrites."""

    # Build structured prompt
    ats_section = ""
    cover_letter_section = ""
    if is_pro:
        ats_section = """
10. "ats_tips": A JSON array of 5-7 specific ATS optimization tips for this resume/job combination (strings).
11. "cover_letter": A professional, personalized cover letter (plain text, 3-4 paragraphs) tailored to this specific job."""
    else:
        ats_section = """
10. "ats_tips": A JSON array of 3 general ATS tips (strings). Note these are limited tips - Pro users get full ATS optimization.
11. "cover_letter": null"""

    user_prompt = f"""Analyze this resume against the job description and provide a comprehensive response.

=== RESUME ===
{resume_text[:6000]}

=== JOB DESCRIPTION ===
{job_description[:3000]}

Return ONLY a valid JSON object with these exact keys:
{{
  "match_score": <integer 0-100 representing how well the resume matches the job>,
  "score_explanation": "<2-3 sentence explanation of the score>",
  "keywords_found": <JSON array of important keywords/skills from the job description that ARE in the resume>,
  "keywords_missing": <JSON array of important keywords/skills from the job description that are MISSING from the resume>,
  "improvements": <JSON array of 4-6 specific improvement suggestions (strings)>,
  "rewritten_resume": "<The complete rewritten resume optimized for this job. Use clean plain text with clear section headers like PROFESSIONAL SUMMARY, EXPERIENCE, EDUCATION, SKILLS. Preserve all factual information but rephrase to match job language, add relevant keywords naturally, strengthen action verbs, and quantify achievements where implied. Keep it to 1-2 pages worth of content.>",
  "strengths": <JSON array of 3-5 things the resume does well for this job>,{ats_section}
}}

Important rules:
- Do NOT fabricate any experience, companies, degrees, or specific metrics not implied by the original
- DO strengthen language, add relevant keywords naturally, improve formatting structure
- Match the job's terminology and preferred phrases
- Ensure keywords_found and keywords_missing are arrays of short strings (1-4 words each)
- Make the rewritten_resume significantly better than the original for this specific job
- Return ONLY the JSON, no other text"""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=4000,
        response_format={"type": "json_object"}
    )

    result_text = response.choices[0].message.content
    result = json.loads(result_text)

    # Validate and sanitize
    result["match_score"] = max(0, min(100, int(result.get("match_score", 50))))
    result["keywords_found"] = result.get("keywords_found", [])[:20]
    result["keywords_missing"] = result.get("keywords_missing", [])[:20]
    result["improvements"] = result.get("improvements", [])
    result["strengths"] = result.get("strengths", [])
    result["ats_tips"] = result.get("ats_tips", [])
    result["cover_letter"] = result.get("cover_letter")

    return result


def generate_pdf_resume(resume_text):
    """Generate a downloadable PDF from the rewritten resume text."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    lines = resume_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue

        # Detect section headers (all caps or ending with colon)
        is_header = (line.isupper() and len(line) > 3) or (line.endswith(':') and len(line) < 40)

        if is_header:
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(30, 64, 175)  # Blue
            pdf.ln(4)
            pdf.cell(0, 8, line, ln=True)
            pdf.set_draw_color(30, 64, 175)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith('•') or line.startswith('-'):
            pdf.set_font("Helvetica", "", 10)
            clean = line.lstrip('•- ').strip()
            pdf.cell(5, 6, "•", ln=False)
            pdf.multi_cell(0, 6, clean)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, line)

    return bytes(pdf.output())


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    pro_token = session.get("pro_token") or request.args.get("token")
    pro_sub = validate_pro_token(pro_token) if pro_token else None
    is_pro = pro_sub is not None

    allowed, remaining = check_free_limit()
    stripe_pub_key = os.getenv("STRIPE_PUBLISHABLE_KEY", "")

    return render_template(
        "index.html",
        is_pro=is_pro,
        pro_email=pro_sub.email if pro_sub else None,
        free_allowed=allowed,
        free_remaining=remaining,
        free_limit=FREE_DAILY_LIMIT,
        stripe_pub_key=stripe_pub_key,
        pro_price_id=STRIPE_PRO_PRICE_ID,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """Main endpoint: receive PDF + job description, return AI analysis."""
    pro_token = session.get("pro_token") or request.form.get("pro_token", "").strip()
    pro_sub = validate_pro_token(pro_token) if pro_token else None
    is_pro = pro_sub is not None

    # Check free limit
    if not is_pro:
        allowed, remaining = check_free_limit()
        if not allowed:
            return jsonify({
                "error": "free_limit_reached",
                "message": f"You've used your {FREE_DAILY_LIMIT} free analysis today. Upgrade to Pro for unlimited rewrites!"
            }), 429

    # Validate inputs
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    resume_file = request.files["resume"]
    job_description = request.form.get("job_description", "").strip()

    if not resume_file or resume_file.filename == "":
        return jsonify({"error": "Please select a PDF resume to upload"}), 400

    if not resume_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    if not job_description or len(job_description) < 50:
        return jsonify({"error": "Please paste a complete job description (at least 50 characters)"}), 400

    if len(job_description) > 8000:
        return jsonify({"error": "Job description is too long. Please trim it to under 8000 characters."}), 400

    # Extract PDF text
    try:
        file_bytes = resume_file.read()
        resume_text = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not resume_text or len(resume_text) < 100:
        return jsonify({"error": "Could not extract enough text from the PDF. Make sure it's a text-based PDF (not a scanned image)."}), 400

    # Call AI
    try:
        result = ai_analyze_resume(resume_text, job_description, is_pro=is_pro)
    except Exception as e:
        app.logger.error(f"OpenAI error: {e}")
        return jsonify({"error": "AI analysis failed. Please try again in a moment."}), 500

    # Save job record
    try:
        job_record = ResumeJob(
            ip_hash=get_ip_hash(),
            pro_token=pro_token if is_pro else None,
            is_pro=is_pro,
            match_score=result["match_score"],
            keywords_found=json.dumps(result["keywords_found"]),
            keywords_missing=json.dumps(result["keywords_missing"]),
            rewritten_resume=result["rewritten_resume"],
            cover_letter=result.get("cover_letter"),
            ats_tips=json.dumps(result.get("ats_tips", [])),
        )
        db.session.add(job_record)
        db.session.commit()
    except Exception as e:
        app.logger.error(f"DB save error: {e}")
        db.session.rollback()

    # Consume free usage
    if not is_pro:
        try:
            consume_free_usage()
        except Exception as e:
            app.logger.error(f"Usage tracking error: {e}")

    # Build response
    response_data = {
        "success": True,
        "job_id": job_record.job_id if job_record else None,
        "match_score": result["match_score"],
        "score_explanation": result.get("score_explanation", ""),
        "keywords_found": result["keywords_found"],
        "keywords_missing": result["keywords_missing"],
        "improvements": result.get("improvements", []),
        "strengths": result.get("strengths", []),
        "rewritten_resume": result["rewritten_resume"],
        "ats_tips": result.get("ats_tips", []),
        "cover_letter": result.get("cover_letter") if is_pro else None,
        "is_pro": is_pro,
    }

    return jsonify(response_data)


@app.route("/download-pdf/<job_id>")
def download_pdf(job_id):
    """Generate and download a PDF of the rewritten resume."""
    job = ResumeJob.query.filter_by(job_id=job_id).first_or_404()

    if not job.rewritten_resume:
        abort(404)

    # Pro check for PDF download
    pro_token = session.get("pro_token") or request.args.get("token", "")
    pro_sub = validate_pro_token(pro_token) if pro_token else None

    if not job.is_pro and not pro_sub:
        # Free users can still download (limited feature)
        pass

    try:
        pdf_bytes = generate_pdf_resume(job.rewritten_resume)
    except Exception as e:
        app.logger.error(f"PDF generation error: {e}")
        abort(500)

    response = app.response_class(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=optimized_resume_{job_id[:8]}.pdf",
            "Content-Length": len(pdf_bytes)
        }
    )
    return response


# ── Stripe / Payments ──────────────────────────────────────────────────────────
@app.route("/create-checkout", methods=["POST"])
def create_checkout():
    """Create Stripe Checkout session for Pro subscription."""
    data = request.get_json() or {}
    email = data.get("email", "").strip()

    if not email or "@" not in email:
        return jsonify({"error": "Valid email required"}), 400

    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=email,
            line_items=[{
                "price": STRIPE_PRO_PRICE_ID,
                "quantity": 1,
            }],
            success_url=f"{APP_URL}/pro-success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{APP_URL}/?canceled=1",
            metadata={"email": email},
            subscription_data={
                "metadata": {"email": email}
            }
        )
        return jsonify({"checkout_url": checkout_session.url})
    except stripe.error.StripeError as e:
        app.logger.error(f"Stripe error: {e}")
        return jsonify({"error": "Payment setup failed. Please try again."}), 500


@app.route("/pro-success")
def pro_success():
    """Handle successful Stripe checkout."""
    session_id = request.args.get("session_id", "")
    if not session_id:
        return redirect(url_for("index"))

    try:
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        email = checkout_session.customer_email or checkout_session.metadata.get("email", "")
        customer_id = checkout_session.customer
        subscription_id = checkout_session.subscription

        # Check if already exists
        existing = ProSubscription.query.filter_by(email=email).first()
        if existing:
            existing.stripe_customer_id = customer_id
            existing.stripe_subscription_id = subscription_id
            existing.status = "active"
            existing.expires_at = None
            token = existing.pro_token
        else:
            token = hashlib.sha256(f"{email}{uuid.uuid4()}".encode()).hexdigest()
            sub = ProSubscription(
                email=email,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                status="active",
                pro_token=token,
            )
            db.session.add(sub)

        db.session.commit()
        session["pro_token"] = token

        return redirect(url_for("index") + f"?pro=1&token={token}")
    except Exception as e:
        app.logger.error(f"Pro success error: {e}")
        return redirect(url_for("index") + "?error=payment_verification_failed")


@app.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhook events."""
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        return jsonify({"error": "Invalid signature"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "customer.subscription.deleted":
        sub_id = data.get("id")
        sub = ProSubscription.query.filter_by(stripe_subscription_id=sub_id).first()
        if sub:
            sub.status = "canceled"
            sub.expires_at = datetime.utcnow()
            db.session.commit()

    elif event_type == "customer.subscription.updated":
        sub_id = data.get("id")
        status = data.get("status")
        sub = ProSubscription.query.filter_by(stripe_subscription_id=sub_id).first()
        if sub:
            sub.status = "active" if status == "active" else status
            db.session.commit()

    elif event_type == "invoice.payment_failed":
        customer_id = data.get("customer")
        sub = ProSubscription.query.filter_by(stripe_customer_id=customer_id).first()
        if sub:
            sub.status = "past_due"
            db.session.commit()

    return jsonify({"received": True})


@app.route("/activate-pro", methods=["POST"])
def activate_pro():
    """Activate Pro with a token (for users returning with token link)."""
    data = request.get_json() or {}
    token = data.get("token", "").strip()
    sub = validate_pro_token(token)
    if sub:
        session["pro_token"] = token
        return jsonify({"success": True, "email": sub.email})
    return jsonify({"error": "Invalid or expired Pro token"}), 400


@app.route("/check-status")
def check_status():
    """Check current user status."""
    pro_token = session.get("pro_token") or request.args.get("token", "")
    pro_sub = validate_pro_token(pro_token) if pro_token else None
    allowed, remaining = check_free_limit()

    return jsonify({
        "is_pro": pro_sub is not None,
        "pro_email": pro_sub.email if pro_sub else None,
        "free_allowed": allowed,
        "free_remaining": remaining,
        "free_limit": FREE_DAILY_LIMIT,
    })


# ── Error Handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Maximum size is {os.getenv('MAX_FILE_SIZE_MB', 10)}MB"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_ENV") == "development", port=5000)
