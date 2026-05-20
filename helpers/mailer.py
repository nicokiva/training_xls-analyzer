"""
helpers/mailer.py — Envío del análisis por email via Gmail SMTP.

Usa la librería smtplib de la standard library (sin dependencias extra).

Para que funcione necesitás una App Password de Google:
  1. Activar verificación en dos pasos en tu cuenta Gmail.
  2. Ir a myaccount.google.com → Seguridad → Contraseñas de aplicaciones.
  3. Crear una para "Mail" → copiar la contraseña de 16 caracteres.
  4. Usar esa contraseña en --email-password (NO la contraseña real de Gmail).
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587  # Puerto TLS estándar de Gmail


def send_analysis(from_email, password, to_email, subject, body):
    """
    Envía el análisis por email usando Gmail SMTP con TLS.

    El cuerpo se manda como texto plano (Markdown legible directamente
    en cualquier cliente de email).

    Args:
        from_email: Dirección Gmail desde donde se manda (ej: "vos@gmail.com").
        password:   App Password de Google de 16 caracteres.
        to_email:   Dirección destino del email.
        subject:    Asunto del email.
        body:       Cuerpo del email (el análisis en Markdown).

    Raises:
        smtplib.SMTPAuthenticationError: Si la contraseña o usuario son incorrectos.
        smtplib.SMTPException: Para cualquier otro error de envío.
    """
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # Conectar, iniciar TLS y autenticar antes de mandar
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
