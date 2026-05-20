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

Uses the smtplib standard library (no extra dependencies).
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587  # Standard Gmail TLS port


def send_analysis(from_email, password, to_email, subject, body):
    """
    Sends the analysis by email using Gmail SMTP with TLS.

    The body is sent as plain text (Markdown readable directly
    in any email client).

    Args:
        from_email: Gmail address to send from (e.g. "you@gmail.com").
        password:   Google App Password (16 characters).
        to_email:   Destination email address.
        subject:    Email subject.
        body:       Email body (the analysis in Markdown).

    Raises:
        smtplib.SMTPAuthenticationError: If the password or user are incorrect.
        smtplib.SMTPException: For any other sending error.
    """
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # Connect, start TLS and authenticate before sending
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
