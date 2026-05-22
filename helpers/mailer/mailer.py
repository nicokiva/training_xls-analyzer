"""
helpers/mailer.py — Sending the analysis by email via Gmail SMTP.

SMTP (Simple Mail Transfer Protocol) is the standard protocol computers use to
send email. Python's smtplib handles the low-level connection for us — we just
call connect, authenticate, and send.

TLS (Transport Layer Security) encrypts the connection so credentials and the
email body can't be read by anyone intercepting the traffic. Gmail requires it:
port 587 accepts plain connections but the server immediately upgrades them to
TLS via the STARTTLS command (server.starttls() below).

Why an App Password instead of your real Gmail password?
  Google blocks "less secure" logins (plain username + password) when two-step
  verification is on. An App Password is a one-off 16-character code that Google
  generates for a specific app — it grants access only to Gmail sending, not to
  your whole account. If it's ever compromised you can revoke just that key.

To get one:
  1. Enable two-step verification on your Gmail account.
  2. Go to myaccount.google.com → Security → App passwords.
  3. Create one for "Mail" → copy the 16-character password.
  4. Use that password in --email-password (NOT your real Gmail password).

Uses smtplib (standard library) and markdown (pip package) to convert the
Markdown body to HTML so bold, headers and lists render properly in email.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown as md

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587  # Standard Gmail TLS port

# Minimal CSS injected into every email so it looks decent in most clients.
_EMAIL_CSS = """
<style>
  body  { font-family: Arial, sans-serif; font-size: 15px; color: #222; line-height: 1.6; max-width: 720px; margin: 0 auto; padding: 20px; }
  h1    { font-size: 1.5em; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
  h2    { font-size: 1.2em; margin-top: 1.4em; }
  h3    { font-size: 1.05em; }
  hr    { border: none; border-top: 1px solid #ddd; margin: 24px 0; }
  code  { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
  ul,ol { padding-left: 1.4em; }
  strong{ color: #111; }
</style>
"""


def _markdown_to_html(text):
    """Converts a Markdown string to a full HTML document with inline CSS."""
    body_html = md.markdown(text, extensions=["extra"])
    return f"<html><head>{_EMAIL_CSS}</head><body>{body_html}</body></html>"


def send_analysis(from_email, password, to_email, subject, body):
    """
    Sends the analysis by email using Gmail SMTP with TLS.

    The body is converted from Markdown to HTML so bold text, headers, and
    bullet lists render properly in Gmail and other email clients.

    Args:
        from_email: Gmail address to send from (e.g. "you@gmail.com").
        password:   Google App Password (16 characters).
        to_email:   Destination email address.
        subject:    Email subject.
        body:       Email body in Markdown format.

    Raises:
        smtplib.SMTPAuthenticationError: If the password or user are incorrect.
        smtplib.SMTPException: For any other sending error.
    """
    html = _markdown_to_html(body)

    msg = MIMEMultipart("alternative")
    msg["From"]    = from_email
    msg["To"]      = to_email
    msg["Subject"] = subject
    # Attach both plain text (fallback) and HTML (preferred).
    # Email clients always use the last part they support — HTML wins.
    msg.attach(MIMEText(body, "plain", "utf-8"))
    msg.attach(MIMEText(html,  "html",  "utf-8"))

    # Connect, start TLS and authenticate before sending
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
