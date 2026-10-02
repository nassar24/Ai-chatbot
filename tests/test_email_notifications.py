"""app.notifications.email tests — smtplib is mocked, no real SMTP
connection is ever made."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.notifications import email as email_module


_ENV = {
    "SMTP_HOST": "smtp.hostinger.com",
    "SMTP_PORT": "465",
    "SMTP_USE_SSL": "true",
    "SMTP_USER": "bot@apexcreative.example",
    "SMTP_PASSWORD": "secret",
    "SMTP_FROM_EMAIL": "bot@apexcreative.example",
    "INTERNAL_NOTIFICATION_EMAIL": "sales@apexcreative.example",
}


def test_missing_config_returns_false_without_raising(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    result = email_module._send("subject", "body")
    assert result is False


def test_missing_recipient_returns_false(monkeypatch):
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("INTERNAL_NOTIFICATION_EMAIL", raising=False)
    result = email_module._send("subject", "body")
    assert result is False


def test_successful_send_uses_ssl_when_configured(monkeypatch):
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)

    mock_server = MagicMock()
    with patch.object(email_module.smtplib, "SMTP_SSL") as mock_smtp_ssl:
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server
        result = email_module._send("Test subject", "Test body")

    assert result is True
    mock_server.login.assert_called_once_with("bot@apexcreative.example", "secret")
    mock_server.send_message.assert_called_once()


def test_smtp_failure_returns_false_not_raise(monkeypatch):
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)

    with patch.object(email_module.smtplib, "SMTP_SSL") as mock_smtp_ssl:
        mock_smtp_ssl.side_effect = OSError("connection refused")
        result = email_module._send("subject", "body")

    assert result is False


def test_send_lead_notification_formats_fields(monkeypatch):
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)

    lead = {
        "name": "Ahmed",
        "email": "ahmed@x.com",
        "phone": None,
        "whatsapp_number": None,
        "company": None,
        "business_type": None,
        "service_requested": "Website",
        "budget": None,
        "timeline": None,
        "notes": None,
    }

    mock_server = MagicMock()
    with patch.object(email_module.smtplib, "SMTP_SSL") as mock_smtp_ssl:
        mock_smtp_ssl.return_value.__enter__.return_value = mock_server
        result = email_module.send_lead_notification(lead, "session-123")

    assert result is True
    sent_message = mock_server.send_message.call_args[0][0]
    body = sent_message.get_content()
    assert "Ahmed" in body
    assert "Website" in body
    assert "session-123" in body
