// Initialisation de la base de données Dexie
const db = new Dexie("LlinkDB");
db.version(2).stores({ chats: '++id, title, time, mode', msgs: '++id, chatId, role, text' });

const SERVER_URL = "https://llink-usz9.onrender.com";
let curChatId = null;
window.isOnline = true; 
let currentMode = 'chat';
window.isLoggedIn = false;
window.isOfflineValid = false;
let chatToRename = null;

// Chargement des préférences au démarrage
document.addEventListener('DOMContentLoaded', () => { 
    const prefInput = document.getElementById('pref-input');
    if (prefInput) {
        prefInput.value = localStorage.getItem('llink_prefs') || ""; 
    }
});

function savePreferences() { 
    localStorage.setItem('llink_prefs', document.getElementById('pref-input').value); 
    alert("Préférences sauvegardées !"); 
    toggleSettingPanel('panel-pref');
}

// Vérification de l'abonnement et de la validité hors-ligne
window.checkSubscription = async () => {
    if(!window.isLoggedIn) return;
    try {
        const res = await fetch(`${SERVER_URL}/api/user/status`, { 
            headers: {"Authorization": `Bearer ${window.authToken}`} 
        });
        const data = await res.json();
        
        if(data.isNewUser) {
            alert("Bienvenue sur Llink ! Vous bénéficiez d'une période d'essai de 3 jours à partir d'aujourd'hui. Il vous faudra vous abonner après cela.");
        }
        
        window.isOfflineValid = data.isValid;
        const infoDiv = document.getElementById('subscription-info');
        
        if(data.isValid && data.isTrial) {
            infoDiv.innerHTML = `🟢 Période d'essai valide jusqu'au : ${new Date(data.expirationDate).toLocaleDateString()}`;
        } else if(data.isValid && !data.isTrial) {
            infoDiv.innerHTML = `💎 Abonnement Pro valide jusqu'au : ${new Date(data.expirationDate).toLocaleDateString()}`;
        } else { 
            infoDiv.innerHTML = `🔴 Accès Hors-Ligne expiré.`; 
            window.isOfflineValid = false; 
        }
        
        updateOfflineUI();
    } catch(e) { 
        console.error("Erreur checkSubscription :", e); 
    }
};

async function submitSubscription() {
    if(!window.isLoggedIn) { alert("Vous devez être connecté pour vous abonner."); return; }
    const txId = document.getElementById('transaction-id').value.trim();
    if(!txId) return;
    
    document.getElementById('subscription-info').innerText = "Vérification en cours...";
    try {
        const res = await fetch(`${SERVER_URL}/api/verify-payment`, {
            method: 'POST', 
            headers: {"Content-Type":"application/json", "Authorization": `Bearer ${window.authToken}`},
            body: JSON.stringify({ transactionId: txId })
        });
        const data = await res.json();
        if(data.success) { 
            alert("Paiement validé avec succès !"); 
            await window.checkSubscription(); 
            document.getElementById('transaction-id').value = ''; 
        } else {
            alert("Erreur : ID de transaction invalide.");
        }
    } catch(e) { 
        alert("Erreur réseau lors de la validation."); 
    }
}

async function syncChatToServer(chat) {
    if(!window.isLoggedIn) return;
    try { 
        await fetch(`${SERVER_URL}/api/history/sync`, { 
            method: "POST", 
            headers: {"Content-Type":"application/json", "Authorization": `Bearer ${window.authToken}`}, 
            body: JSON.stringify(chat) 
        }); 
    } catch(e) {
        console.error("Erreur de synchronisation du chat:", e);
    }
}

window.restoreHistory = async () => {
    try {
        const user = window.auth ? window.auth.currentUser : null;
        if (!user) return;
        const userToken = await user.getIdToken();
        const response = await fetch(`${SERVER_URL}/api/history`, { 
            method: 'GET', 
            headers: { 'Authorization': `Bearer ${userToken}`, 'Content-Type': 'application/json' } 
        });
        if (!response.ok) return;
        const data = await response.json();
        if (!data.warning) console.log("Historique distant synchronisé avec succès.");
    } catch (error) { 
        console.error("Erreur lors de la restauration de l'historique:", error); 
    }
};

function updateOfflineUI() {
    const alertBox = document.getElementById('pro-alert');
    if(!window.isOnline && !window.isOfflineValid) {
        alertBox.classList.remove('llink-hidden');
    } else {
        alertBox.classList.add('llink-hidden');
    }
}

window.toggleOnline = () => {
    window.isOnline = !window.isOnline;
    const dot = document.getElementById('network-status');
    const modelSel = document.getElementById('model-selector');
    const micBtn = document.getElementById('mic-btn');
    
    if(window.isOnline) {
        dot.className = "llink-status-online";
        modelSel.disabled = false; 
        modelSel.style.opacity = 1;
        micBtn.disabled = false; 
        micBtn.style.opacity = 1;
    } else {
        dot.className = "llink-status-offline";
        modelSel.disabled = true; 
        modelSel.style.opacity = 0.5;
        micBtn.disabled = true; 
        micBtn.style.opacity = 0.5;
        if(!window.isLoggedIn) alert("Vous devez être connecté pour vérifier l'accès au mode Hors-Ligne.");
    }
    updateOfflineUI();
};

// Fonctions d'interface utilisateur
function toggleTheme() { document.documentElement.classList.toggle('dark-theme'); }
function openSettings() { 
    document.getElementById('settings-modal').style.display = 'flex'; 
    if(window.innerWidth < 768) toggleSidebar(); 
}
function closeSettings() { document.getElementById('settings-modal').style.display = 'none'; }
function toggleSettingPanel(id) { 
    const p = document.getElementById(id); 
    p.style.display = p.style.display === 'block' ? 'none' : 'block'; 
}
function toggleSidebar() { document.getElementById('sidebar').classList.toggle('llink-sidebar-hidden'); }
function toggleMediaMenu() { 
    const m = document.getElementById('media-menu'); 
    m.style.display = m.style.display === 'flex' ? 'none' : 'flex'; 
}
function triggerInput(id) { document.getElementById(id).click(); toggleMediaMenu(); }
function scrollToBottom() { 
    const w = document.getElementById('chat-window'); 
    w.scrollTop = w.scrollHeight; 
}
window.closeModal = (id) => { document.getElementById(id).style.display = 'none'; };

async function toggleMode() {
    currentMode = currentMode === 'chat' ? 'translate' : 'chat';
    const btn = document.getElementById('mode-btn'); 
    const lang = document.getElementById('lang-bar');
    
    if(currentMode === 'chat') { 
        btn.innerHTML = '<i data-lucide="message-square" class="llink-icon-standard"></i>'; 
        lang.classList.add('llink-hidden'); 
        lang.classList.remove('llink-flex'); 
    } else { 
        btn.innerHTML = '<i data-lucide="languages" class="llink-icon-active"></i>'; 
        lang.classList.remove('llink-hidden'); 
        lang.classList.add('llink-flex'); 
    }
    lucide.createIcons(); 
    await newConversation(); 
    window.loadHistory();
}

window.newConversation = () => {
    curChatId = null;
    const w = document.getElementById('chat-window');
    
    if(currentMode === 'translate') {
        w.innerHTML = `
            <div id="empty-state" class="llink-empty-state">
                <div class="llink-logo-container"><svg class="llink-logo-svg" viewBox="-5 -5 110 110"><circle cx="50" cy="20" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="76" cy="35" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="76" cy="65" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="50" cy="80" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="24" cy="65" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="24" cy="35" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/></svg></div>
                <h1 class="llink-title-gradient">Llink Traduction</h1>
                <p class="llink-subtitle">Posez vos questions ou lancez une traduction.</p>
            </div>`;
    } else {
        w.innerHTML = `
            <div id="empty-state" class="llink-empty-state">
                <div class="llink-logo-container"><svg width="100" height="100" viewBox="-20 0 100 100"><circle cx="50" cy="20" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="76" cy="35" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="76" cy="65" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="50" cy="80" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="24" cy="65" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="24" cy="35" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/></svg></div>
                <h1 class="llink-title-gradient">Llink</h1>
                <p class="llink-subtitle">Posez vos questions ou lancez une traduction.</p>
                <div class="llink-suggestions-grid">
                    <button onclick="sendDirect('M\\'aider à apprendre')" class="llink-suggestion-btn">
                        <span class="llink-suggestion-title llink-text-blue">🎓 M'aider à apprendre</span>
                        <span class="llink-suggestion-desc">Générer des explications méthodiques pour vos cours avec des exercices adaptés.</span>
                    </button>
                    <button onclick="sendDirect('M\\'aider à rédiger')" class="llink-suggestion-btn">
                        <span class="llink-suggestion-title llink-text-green">✍️ M'aider à rédiger</span>
                        <span class="llink-suggestion-desc">Perfectionnez vos lettres, codes sources et autres documents écrits.</span>
                    </button>
                    <button onclick="sendDirect('Discuter de quelque chose')" class="llink-suggestion-btn">
                        <span class="llink-suggestion-title llink-text-purple">💬 Discuter de quelque chose</span>
                        <span class="llink-suggestion-desc">Explorer des thématiques et des sujets qui vous passionnent.</span>
                    </button>
                    <button onclick="sendDirect('Me surprendre')" class="llink-suggestion-btn">
                        <span class="llink-suggestion-title llink-text-orange">✨ Me surprendre</span>
                        <span class="llink-suggestion-desc">Laissez Llink vous surprendre avec quelque chose d'inattendu.</span>
                    </button>
                    <button onclick="sendDirect('Comment utiliser Llink ?')" class="llink-suggestion-btn llink-col-span-full">
                        <span class="llink-suggestion-title llink-text-darkblue">🚀 Comment utiliser</span>
                        <span class="llink-suggestion-desc">Comprendre en quoi Llink diffère des IA actuelles et comment l'utiliser.</span>
                    </button>
                </div>
            </div>`;
    }
};

window.loadHistory = async () => {
    if(!window.isLoggedIn) return;
    const h = await db.chats.where('mode').equals(currentMode).reverse().toArray();
    document.getElementById('history').innerHTML = h.map(c => 
        `<div class="llink-history-item">
            <div class="llink-history-title" onclick="switchChat(${c.id})">${c.title}</div>
            <div class="llink-history-actions">
                <i data-lucide="edit-2" class="llink-icon-edit" onclick="openRenameModal(${c.id}, \`${c.title.replace(/`/g, "")}\`, event)"></i>
                <i data-lucide="trash-2" class="llink-icon-delete" onclick="deleteChat(${c.id}, event)"></i>
            </div>
        </div>`
    ).join('');
    lucide.createIcons();
};

window.openRenameModal = (id, currentTitle, event) => {
    if(event) event.stopPropagation();
    chatToRename = id;
    document.getElementById('rename-input').value = currentTitle;
    document.getElementById('rename-modal').style.display = 'flex';
};

window.saveChatName = async () => {
    const newTitle = document.getElementById('rename-input').value.trim();
    if (newTitle && chatToRename) {
        await db.chats.update(chatToRename, { title: newTitle });
        window.loadHistory();
        closeModal('rename-modal');
        const fullChat = await db.chats.get(chatToRename);
        fullChat.messages = await db.msgs.where('chatId').equals(chatToRename).toArray();
        await syncChatToServer(fullChat);
    }
};

window.switchChat = async (id) => {
    curChatId = id;
    const m = await db.msgs.where('chatId').equals(id).toArray();
    document.getElementById('chat-window').innerHTML = "";
    m.forEach(msg => renderMessage(msg.text, msg.role));
    if(window.innerWidth < 768) toggleSidebar();
};

window.deleteChat = async (id, event) => { 
    if(event) event.stopPropagation();
    if(confirm("Supprimer la discussion ?")) { 
        await db.chats.delete(id);
        await db.msgs.where('chatId').equals(id).delete();
        if(curChatId === id) newConversation(); 
        window.loadHistory(); 
    } 
};

function parseMarkdown(text) {
    if(!text) return "";
    let html = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    html = html.replace(/```([\s\S]*?)```/g, function(m, p1){ 
        return `<div class="llink-code-block"><button onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText); this.innerText='Copié!'; setTimeout(()=>this.innerText='Copier',2000);" class="llink-copy-code-btn">Copier</button><pre><code>${p1}</code></pre></div>`; 
    });
    html = html.replace(/\*\*([\s\S]*?)\*\*/g, '<strong>$1</strong>'); 
    html = html.replace(/\n/g, '<br>'); 
    return html;
}

// ================= INJECTION DES MESSAGES =================
function renderMessage(txt, role, id="") {
    const w = document.getElementById('chat-window');
    const html = parseMarkdown(txt);
    const safeTxt = encodeURIComponent(txt);
    
    const actionBtns = `
        <div class="llink-msg-actions">
            <button onclick="navigator.clipboard.writeText(decodeURIComponent('${safeTxt}')); this.innerHTML='<i data-lucide=\\'check\\' class=\\'llink-icon-success\\'></i>'; setTimeout(() => { this.innerHTML='<i data-lucide=\\'copy\\' class=\\'llink-icon-standard\\'></i>'; lucide.createIcons(); }, 2000);" title="Copier" class="llink-action-btn"><i data-lucide="copy" class="llink-icon-standard"></i></button>
            <button onclick="if(navigator.share) { navigator.share({title: 'Llink', text: decodeURIComponent('${safeTxt}')}) } else { alert('Partage non supporté.'); }" title="Partager" class="llink-action-btn"><i data-lucide="share-2" class="llink-icon-standard"></i></button>
        </div>
    `;

    if(role === 'user') {
        w.insertAdjacentHTML('beforeend', `
            <div class="llink-msg-wrapper llink-align-right" id="${id}">
                <div class="llink-msg-bubble-user msg-content">${html}</div>
                ${actionBtns}
            </div>`);
    } else {
        w.insertAdjacentHTML('beforeend', `
            <div class="llink-msg-wrapper llink-align-left" id="${id}">
                <div class="llink-bot-header">
                    <div class="llink-bot-avatar">
                        <svg viewBox="-5 -5 110 110" class="llink-logo-svg"><circle cx="50" cy="20" r="18" fill="none" stroke="#FF5733" stroke-width="8"/><circle cx="76" cy="35" r="18" fill="none" stroke="#33FF57" stroke-width="8"/><circle cx="76" cy="65" r="18" fill="none" stroke="#3357FF" stroke-width="8"/><circle cx="50" cy="80" r="18" fill="none" stroke="#F333FF" stroke-width="8"/><circle cx="24" cy="65" r="18" fill="none" stroke="#FFBD33" stroke-width="8"/><circle cx="24" cy="35" r="18" fill="none" stroke="#33FFF3" stroke-width="8"/></svg>
                    </div>
                    <span class="llink-bot-name">LLINK</span>
                </div>
                <div class="llink-msg-bubble-bot msg-content">${html}</div>
                ${actionBtns}
            </div>`);
    }
    lucide.createIcons(); 
    scrollToBottom();
}

window.sendDirect = (txt) => { 
    document.getElementById('user-msg').value = txt; 
    sendFromInput(); 
};

window.sendFromInput = async () => {
    const input = document.getElementById('user-msg');
    const txt = input.value;
    if(!txt.trim()) return; 
    
    input.value = ''; 
    input.style.height = 'auto'; // Reset de la hauteur après envoi
    
    if(!curChatId) { 
        curChatId = await db.chats.add({title: txt.substring(0,20), time: Date.now(), mode: currentMode}); 
        window.loadHistory(); 
    }
    
    const emptyState = document.getElementById('empty-state');
    if(emptyState) emptyState.remove();
    
    if(window.isLoggedIn) await db.msgs.add({chatId: curChatId, role: 'user', text: txt});
    renderMessage(txt, 'user');
    
    const loadId = "load-" + Date.now(); 
    renderMessage("Réflexion...", 'bot', loadId);
    const contentDiv = document.getElementById(loadId).querySelector('.msg-content');
    let fullReply = "";

    // ================= LOGIQUE EN LIGNE / HORS-LIGNE COMPLÉTÉE =================
    if(window.isOnline) {
        try {
            const prefs = localStorage.getItem('llink_prefs') || "";
            const payload = { message: txt, mode: currentMode, model: document.getElementById('model-selector').value, preferences: prefs };
            const res = await fetch(`${SERVER_URL}/api/chat`, { 
                method: 'POST', 
                body: JSON.stringify(payload), 
                headers: {"Content-Type":"application/json", "Authorization": window.authToken ? `Bearer ${window.authToken}` : ""} 
            });
            const reader = res.body.getReader();
            const decoder = new TextDecoder(); 
            contentDiv.innerHTML = "";
            
            while(true) { 
                const {value, done} = await reader.read(); 
                if(done) break;
                fullReply += decoder.decode(value, {stream:true}); 
                contentDiv.innerHTML = parseMarkdown(fullReply); 
                scrollToBottom(); 
            }
        } catch(e) { 
            fullReply = "❌ Erreur de connexion au serveur Render."; 
            contentDiv.innerHTML = fullReply; 
        }
    } else {
        // --- LOGIQUE HORS-LIGNE COMPLÈTE ---
        if(!window.isOfflineValid) {
            fullReply = "❌ Votre abonnement pro a expiré. Mode hors-ligne indisponible.";
            contentDiv.innerHTML = fullReply;
        } else if(window.localAI) {
            try {
                contentDiv.innerHTML = "Génération locale en cours...";
                const messages = [{ role: "user", content: txt }];
                
                // Exécution du modèle ONNX (Transformers.js)
                const output = await window.localAI(messages, {
                    max_new_tokens: 256,
                    temperature: 0.7
                });
                
                fullReply = output[0].generated_text[output[0].generated_text.length - 1].content || output[0].generated_text;
                contentDiv.innerHTML = parseMarkdown(fullReply);
                scrollToBottom();
            } catch(e) {
                console.error(e);
                fullReply = "❌ Erreur critique lors de la génération avec le modèle local.";
                contentDiv.innerHTML = fullReply;
            }
        } else {
            fullReply = "⏳ Le modèle d'intelligence artificielle local est toujours en cours de chargement en arrière-plan. Veuillez patienter.";
            contentDiv.innerHTML = fullReply;
        }
    }
    
    // Sauvegarde du message bot généré
    if(window.isLoggedIn && fullReply) {
        await db.msgs.add({chatId: curChatId, role: 'bot', text: fullReply});
    }
};

        // ================= MEDIA LOGIC =================
        let isRecording = false;
        let mediaRecorder;
        let audioChunks = [];
        window.handleImageUpload = async (e) => {
            const f = e.target.files[0];
            if(!f) return; toggleMediaMenu();
            if(!window.isOnline) {
                renderMessage("OCR Local...", 'bot', 'ocr-load');
                const {data:{text}} = await Tesseract.recognize(f, 'fra+eng');
                document.getElementById('ocr-load').remove(); sendDirect("Analyse cette image: " + text); return;
            }
            const reader = new FileReader(); reader.readAsDataURL(f);
            reader.onload = async () => {
                renderMessage("Analyse d'image Cloud...", 'bot', 'ocr-load');
                try {
                    const res = await fetch(`${SERVER_URL}/api/ocr`, { method: 'POST', body: JSON.stringify({imageBase64: reader.result}), headers: {"Content-Type":"application/json"} });
                    const data = await res.json(); document.getElementById('ocr-load').remove(); sendDirect("Analyse: " + data.response);
                } catch(e) { document.getElementById('ocr-load').remove(); alert("Erreur OCR"); }
            }; e.target.value = '';
        };

        window.handleVoice = async () => {
            if(!window.isOnline) { alert("Le micro est désactivé hors-ligne."); return; }
            if(!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
                    mediaRecorder = new MediaRecorder(stream); audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        const blob = new Blob(audioChunks, {type:'audio/wav'});
                        const form = new FormData(); form.append("file", blob);
                        try { const res = await fetch(`${SERVER_URL}/api/audio`, {method:'POST', body:form});
                        const data = await res.json(); if(data.text) sendDirect(data.text); } catch(e){ alert("Erreur serveur audio."); }
                    };
                    mediaRecorder.start(); isRecording = true; document.getElementById('mic-btn').classList.add('text-red-500', 'animate-pulse');
                } catch(e) { alert("Refus d'accès micro."); }
            } else { mediaRecorder.stop(); isRecording = false; document.getElementById('mic-btn').classList.remove('text-red-500', 'animate-pulse'); }
        };

        document.getElementById('file-in').addEventListener('change', handleImageUpload);
        document.getElementById('camera-in').addEventListener('change', handleImageUpload);

        lucide.createIcons(); 
        newConversation();
