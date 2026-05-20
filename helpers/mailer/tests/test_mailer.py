"""
Tests for helpers/mailer/mailer.py — mocks smtplib.SMTP to avoid real network I/O.
"""
import pytest
from unittest.mock import patch, MagicMock, call
from helpers.mailer import send_analysis


FAKE_FROM  = "from@example.com"
FAKE_PASS  = "fakepassword123"
FAKE_TO    = "to@example.com"
FAKE_SUBJ  = "Análisis semanal"
FAKE_BODY  = "## Análisis\n\nTodo bien esta semana."


@pytest.fixture
def smtp_mock():
    """Patch smtplib.SMTP and return the mock instance (context manager)."""
    with patch("helpers.mailer.mailer.smtplib.SMTP") as MockSMTP:
        instance = MagicMock()
        MockSMTP.return_value.__enter__ = MagicMock(return_value=instance)
        MockSMTP.return_value.__exit__  = MagicMock(return_value=False)
        yield MockSMTP, instance


class TestSendAnalysis:
    def test_smtp_connected_to_gmail(self, smtp_mock):
        MockSMTP, _ = smtp_mock
        send_analysis(FAKE_FROM, FAKE_PASS, FAKE_TO, FAKE_SUBJ, FAKE_BODY)
        MockSMTP.assert_called_once_with("smtp.gmail.com", 587)

    def test_starttls_called(self, smtp_mock):
        _, instance = smtp_mock
        send_analysis(FAKE_FROM, FAKE_PASS, FAKE_TO, FAKE_SUBJ, FAKE_BODY)
        instance.starttls.assert_called_once()

    def test_login_called_with_correct_credentials(self, smtp_mock):
        _, instance = smtp_mock
        send_analysis(FAKE_FROM, FAKE_PASS, FAKE_TO, FAKE_SUBJ, FAKE_BODY)
        instance.login.assert_called_once_with(FAKE_FROM, FAKE_PASS)

    def test_sendmail_called_with_correct_addresses(self, smtp_mock):
        _, instance = smtp_mock
        send_analysis(FAKE_FROM, FAKE_PASS, FAKE_TO, FAKE_SUBJ, FAKE_BODY)
        args = instance.sendmail.call_args
        assert args[0][0] == FAKE_FROM
        assert args[0][1] == FAKE_TO

    def test_sendmail_message_contains_subject(self, smtp_mock):
        import email
        from email.header import decode_header
        _, instance = smtp_mock
        send_analysis(FAKE_FROM, FAKE_PASS, FAKE_TO, FAKE_SUBJ, FAKE_BODY)
        raw_message = instance.sendmail.call_args[0][2]
        msg = email.message_from_string(raw_message)
        # Subject may be RFC2047-encoded; decode it
        decoded_subject = ""
        for part, charset in decode_header(msg["Subject"]):
            if isinstance(part, bytes):
                decoded_subject += part.decode(charset or "utf-8")
            else:
                decoded_subject += part
        assert FAKE_SUBJ in decoded_subject

    def test_sendmail_message_contains_body(self, smtp_mock):
        import email
        _, instance = smtp_mock
        send_analysis(FAKE_FROM, FAKE_PASS, FAKE_TO, FAKE_SUBJ, FAKE_BODY)
        raw_message = instance.sendmail.call_args[0][2]
        msg = email.message_from_string(raw_message)
        # Walk all parts to find the plain text body (may be base64-encoded)
        full_body = ""
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    full_body += payload.decode(part.get_content_charset() or "utf-8")
        assert FAKE_BODY in full_body

    def test_order_starttls_before_login(self, smtp_mock):
        """starttls must happen before login."""
        _, instance = smtp_mock
        call_order = []
        instance.starttls.side_effect = lambda: call_order.append("starttls")
        instance.login.side_effect    = lambda *a: call_order.append("login")
        send_analysis(FAKE_FROM, FAKE_PASS, FAKE_TO, FAKE_SUBJ, FAKE_BODY)
        assert call_order.index("starttls") < call_order.index("login")
