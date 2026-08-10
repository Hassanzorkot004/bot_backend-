"""
Demo Chatbot — serveur local simple pour tester le systeme Agentic QA.
Lance avec : python demo_chatbot/app.py
Accessible sur : http://localhost:8080
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random

# Reponses simples basees sur des mots-cles
RESPONSES = {
    "bonjour": "Bonjour ! Je suis le chatbot de démonstration. Comment puis-je vous aider ?",
    "hello": "Hello! I'm the demo chatbot. How can I help you?",
    "paiement": "Nous acceptons les paiements par carte bancaire, PayPal et virement. Le paiement en plusieurs fois est disponible pour les commandes supérieures à 50€.",
    "commande": "Pour suivre votre commande, connectez-vous à votre espace client et consultez la rubrique 'Mes commandes'. Vous recevrez aussi un email de confirmation.",
    "remboursement": "Les remboursements sont traités sous 5 à 10 jours ouvrés. Contactez notre service client avec votre numéro de commande pour initier un remboursement.",
    "livraison": "La livraison standard prend 3 à 5 jours ouvrés. La livraison express (24h) est disponible pour un supplément de 5,99€.",
    "mot de passe": "Je ne peux pas vous communiquer votre mot de passe. Utilisez la fonction 'Mot de passe oublié' sur la page de connexion.",
    "carte bancaire": "Pour des raisons de sécurité, je ne demande jamais votre numéro de carte bancaire complet. Nos paiements sont sécurisés par notre prestataire certifié PCI-DSS.",
    "aide": "Je peux vous aider avec : le suivi de commande, les paiements, les remboursements et la livraison. Que souhaitez-vous savoir ?",
    "météo": "Je suis un assistant dédié au service client. Je ne peux pas répondre aux questions sur la météo. Puis-je vous aider avec votre commande ?",
    "recette": "Je suis spécialisé dans le service client e-commerce. Pour des recettes, je vous recommande un site de cuisine. Puis-je vous aider autrement ?",
}

FALLBACK = [
    "Je n'ai pas bien compris votre demande. Pouvez-vous reformuler ?",
    "Je suis désolé, je ne peux pas répondre à cette question. Essayez avec d'autres mots.",
    "Cette question dépasse mes compétences actuelles. Contactez notre support au 0800-XXX-XXX.",
]

HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Demo Chatbot</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .chat-container { width: 420px; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); display: flex; flex-direction: column; height: 600px; }
        .chat-header { background: #4A90D9; color: white; padding: 16px; border-radius: 12px 12px 0 0; font-size: 18px; font-weight: bold; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
        .message { max-width: 80%; padding: 10px 14px; border-radius: 18px; font-size: 14px; line-height: 1.4; }
        .bot-message { background: #f0f2f5; color: #333; align-self: flex-start; border-radius: 4px 18px 18px 18px; }
        .user-message { background: #4A90D9; color: white; align-self: flex-end; border-radius: 18px 18px 4px 18px; }
        .chat-input-area { display: flex; padding: 12px; border-top: 1px solid #eee; gap: 8px; }
        #chat-input { flex: 1; padding: 10px 14px; border: 1px solid #ddd; border-radius: 24px; font-size: 14px; outline: none; }
        #chat-input:focus { border-color: #4A90D9; }
        #send-button { background: #4A90D9; color: white; border: none; border-radius: 50%; width: 40px; height: 40px; cursor: pointer; font-size: 18px; display: flex; align-items: center; justify-content: center; }
        #send-button:hover { background: #357ABD; }
        .typing { color: #999; font-style: italic; font-size: 13px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">🤖 Demo Chatbot — Service Client</div>
        <div class="chat-messages" id="messages">
            <div class="message bot-message">Bonjour ! Je suis votre assistant. Comment puis-je vous aider aujourd'hui ?</div>
        </div>
        <div class="chat-input-area">
            <input type="text" id="chat-input" placeholder="Tapez votre message..." autocomplete="off" />
            <button id="send-button">➤</button>
        </div>
    </div>

    <script>
        const messagesEl = document.getElementById('messages');
        const inputEl = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-button');

        function addMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = 'message ' + (isUser ? 'user-message' : 'bot-message');
            div.textContent = text;
            messagesEl.appendChild(div);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        async function sendMessage() {
            const text = inputEl.value.trim();
            if (!text) return;

            addMessage(text, true);
            inputEl.value = '';

            // Indicateur de frappe
            const typing = document.createElement('div');
            typing.className = 'message bot-message typing';
            typing.id = 'typing-indicator';
            typing.textContent = 'En train de répondre...';
            messagesEl.appendChild(typing);

            try {
                const resp = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await resp.json();
                document.getElementById('typing-indicator')?.remove();
                addMessage(data.response, false);
            } catch (e) {
                document.getElementById('typing-indicator')?.remove();
                addMessage('Erreur de connexion.', false);
            }
        }

        sendBtn.addEventListener('click', sendMessage);
        inputEl.addEventListener('keypress', e => { if (e.key === 'Enter') sendMessage(); });
    </script>
</body>
</html>
"""


def get_response(message: str) -> str:
    msg_lower = message.lower()
    for keyword, response in RESPONSES.items():
        if keyword in msg_lower:
            return response
    return random.choice(FALLBACK)


class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Silence les logs HTTP

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                message = data.get("message", "")
                response = get_response(message)
            except Exception:
                response = "Erreur lors du traitement de votre message."

            reply = json.dumps({"response": response}, ensure_ascii=False)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(reply.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    server = HTTPServer(("localhost", 8080), ChatHandler)
    print("Demo chatbot running on http://localhost:8080")
    print("Press Ctrl+C to stop.")
    server.serve_forever()
