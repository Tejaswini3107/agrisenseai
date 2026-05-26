import os
import smtplib
from email.message import EmailMessage

SUPPORTED_ALERT_LANGUAGES = {
	"english": {
		"title": "Emergency alert for {crop}",
		"body": "{location}: {detail}. Take action immediately.",
		"subject": "Emergency alert for {crop}",
		"email_body": "Emergency alert for {crop} at {location}. {detail}\n\nTake immediate action and stay safe.",
	},
	"hindi": {
		"title": "{crop} के लिए आपातकालीन अलर्ट",
		"body": "{location}: {detail}. कृपया तुरंत कार्रवाई करें।",
		"subject": "{crop} के लिए आपातकालीन अलर्ट",
		"email_body": "{location} में {crop} के लिए आपातकालीन अलर्ट। {detail}\n\nकृपया तुरंत कार्रवाई करें और सुरक्षित रहें।",
	},
	"telugu": {
		"title": "{crop} కోసం అత్యవసర హెచ్చరిక",
		"body": "{location}: {detail}. దయచేసి వెంటనే చర్య తీసుకోండి.",
		"subject": "{crop} కోసం అత్యవసర హెచ్చరిక",
		"email_body": "{location} వద్ద {crop} కోసం అత్యవసర హెచ్చరిక. {detail}\n\nదయచేసి వెంటనే చర్య తీసుకోండి మరియు సురక్షితంగా ఉండండి.",
	},
	"marathi": {
		"title": "{crop} साठी आपत्कालीन इशारा",
		"body": "{location}: {detail}. कृपया त्वरित कारवाई करा.",
		"subject": "{crop} साठी आपत्कालीन इशारा",
		"email_body": "{location} येथे {crop} साठी आपत्कालीन इशारा. {detail}\n\nकृपया त्वरित कारवाई करा आणि सुरक्षित रहा.",
	},
	"urdu": {
		"title": "{crop} کے لیے ہنگامی الرٹ",
		"body": "{location}: {detail}. براہ کرم فوری کارروائی کریں۔",
		"subject": "{crop} کے لیے ہنگامی الرٹ",
		"email_body": "{location} میں {crop} کے لیے ہنگامی الرٹ۔ {detail}\n\nبراہ کرم فوری کارروائی کریں اور محفوظ رہیں۔",
	},
	"arabic": {
		"title": "تنبيه طارئ لـ {crop}",
		"body": "{location}: {detail}. يرجى اتخاذ إجراء فوري.",
		"subject": "تنبيه طارئ لـ {crop}",
		"email_body": "تنبيه طارئ لـ {crop} في {location}. {detail}\n\nيرجى اتخاذ إجراء فوري والبقاء بأمان.",
	},
	"french": {
		"title": "Alerte d'urgence pour {crop}",
		"body": "{location} : {detail}. Agissez immédiatement.",
		"subject": "Alerte d'urgence pour {crop}",
		"email_body": "Alerte d'urgence pour {crop} à {location}. {detail}\n\nAgissez immédiatement et restez en sécurité.",
	},
}


def normalize_language(language: str | None) -> str:
	value = (language or "english").strip().lower()
	return value if value in SUPPORTED_ALERT_LANGUAGES else "english"


def format_emergency_alert(
	*,
	language: str | None,
	crop: str,
	location: str,
	detail: str,
	severity: str,
) -> dict[str, str]:
	normalized_language = normalize_language(language)
	template = SUPPORTED_ALERT_LANGUAGES[normalized_language]
	values = {
		"crop": crop,
		"location": location,
		"detail": detail,
		"severity": severity,
	}
	return {
		"language": normalized_language,
		"title": template["title"].format(**values),
		"body": template["body"].format(**values),
		"subject": template["subject"].format(**values),
		"email_body": template["email_body"].format(**values),
	}


def is_email_configured() -> bool:
	return all(
		os.getenv(name)
		for name in [
			"SMTP_HOST",
			"SMTP_USERNAME",
			"SMTP_PASSWORD",
			"SMTP_FROM_EMAIL",
		]
	)


def send_email_message(recipients: list[str], subject: str, body: str) -> None:
	if not recipients:
		return
	if not is_email_configured():
		raise RuntimeError("SMTP is not configured")

	host = os.getenv("SMTP_HOST", "")
	port = int(os.getenv("SMTP_PORT", "587"))
	username = os.getenv("SMTP_USERNAME", "")
	password = os.getenv("SMTP_PASSWORD", "")
	from_email = os.getenv("SMTP_FROM_EMAIL", username)
	use_tls = os.getenv("SMTP_USE_TLS", "true").lower() != "false"

	message = EmailMessage()
	message["From"] = from_email
	message["To"] = ", ".join(recipients)
	message["Subject"] = subject
	message.set_content(body)

	with smtplib.SMTP(host, port, timeout=15) as smtp:
		if use_tls:
			smtp.starttls()
		smtp.login(username, password)
		smtp.send_message(message)