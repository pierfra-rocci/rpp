import os

# SMTP settings
SMTP_SERVER = "smtp.mail.ovh.net"
SMTP_PORT = 587
SMTP_USER = "rpp_support@saf-astronomie.fr"
SMTP_PASS = os.getenv("SMTP_PASS")  # Load password from environment variable
