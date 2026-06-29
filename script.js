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
    const prefInput = document.getElementById('pref-input');
    if (prefInput) {
        localStorage.setItem('llink_prefs', prefInput.value); 
        alert("Préférences sauvegardées !"); 
        toggleSettingPanel('panel-pref');
    }
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
        
        if (infoDiv) {
            if(data.isValid && data.isTrial) {
                infoDiv.innerHTML = `🟢 Période d'essai valide jusqu'au : ${new Date(data.expirationDate).toLocaleDateString()}`;
            } else if(data.isValid && !data.isTrial) {
                infoDiv.innerHTML = `💎 Abonnement Pro valide jusqu'au : ${new Date(data.expirationDate).toLocaleDateString()}`;
            } else { 
                infoDiv.innerHTML = `🔴 Accès Hors-Ligne expiré.`; 
                window.isOfflineValid = false; 
            }
        }
        
        updateOfflineUI();
    } catch(e) { 
        console.error("Erreur checkSubscription :", e); 
    }
};

async function submitSubscription() {
    if(!window.isLoggedIn) { alert("Vous devez être connecté pour vous abonner."); return; }
    const txIdInput = document.getElementById('transaction-id');
    if(!txIdInput) return;
    const txId = txIdInput.value.trim();
    if(!txId) return;
    
    const infoDiv = document.getElementById('subscription-info');
    if (infoDiv) infoDiv.innerText = "Vérification en cours...";
    
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
            txIdInput.value = ''; 
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
    if (alertBox) {
        if(!window.isOnline && !window.isOfflineValid) {
            alertBox.classList.remove('hidden');
        } else {
            alertBox.classList.add('hidden');
        }
    }
}

window.toggleOnline = () => {
    window.isOnline = !window.isOnline;
    const dot = document.getElementById('network-status');
    const modelSel = document.getElementById('model-selector');
    const micBtn = document.getElementById('mic-btn');
    
    if(window.isOnline) {
        if(dot) dot.className = "status-dot bg-green-500";
        if(modelSel) { modelSel.disabled = false; modelSel.style.opacity = 1; }
        if(micBtn) { micBtn.disabled = false; micBtn.style.opacity = 1; }
    } else {
        if(dot) dot.className = "status-dot bg-red-500";
        if(modelSel) { modelSel.disabled = true; modelSel.style.opacity = 0.5; }
        if(micBtn) { micBtn.disabled = true; micBtn.style.opacity = 0.5; }
        if(!window.isLoggedIn) alert("Vous devez être connecté pour vérifier l'accès au mode Hors-Ligne.");
    }
    updateOfflineUI();
};

// Fonctions d'interface utilisateur
function toggleTheme() { document.documentElement.classList.toggle('dark'); }
function openSettings() { 
    const modal = document.getElementById('settings-modal');
    if(modal) modal.style.display = 'flex'; 
    if(window.innerWidth < 1024) toggleSidebar(); 
}
function closeSettings() { 
    const modal = document.getElementById('settings-modal');
    if(modal) modal.style.display = 'none'; 
}
function toggleSettingPanel(id) { 
    const p = document.getElementById(id); 
    if(p) p.style.display = p.style.display === 'block' ? 'none' : 'block'; 
}
function toggleSidebar() { document.getElementById('sidebar').classList.toggle('open'); }
function toggleMediaMenu() { 
    const m = document.getElementById('media-menu'); 
    if(m) m.style.display = m.style.display === 'flex' ? 'none' : 'flex'; 
}
function triggerInput(id) { 
    const input = document.getElementById(id);
    if(input) input.click(); 
    toggleMediaMenu(); 
}
function scrollToBottom() { 
    const w = document.getElementById('chat-window'); 
    if(w) w.scrollTop = w.scrollHeight; 
}
window.closeModal = (id) => { 
    const modal = document.getElementById(id);
    if(modal) modal.style.display = 'none'; 
};

async function toggleMode() {
    currentMode = currentMode === 'chat' ? 'translate' : 'chat';
    const btn = document.getElementById('mode-btn'); 
    const lang = document.getElementById('lang-bar');
    
    if(currentMode === 'chat') { 
        if(btn) btn.innerHTML = '<i data-lucide="message-square"></i>'; 
        if(lang) { lang.classList.add('hidden'); lang.classList.remove('flex'); }
    } else { 
        if(btn) btn.innerHTML = '<i data-lucide="languages" class="text-blue-500"></i>'; 
        if(lang) { lang.classList.remove('hidden'); lang.classList.add('flex'); }
    }
    if(typeof lucide !== 'undefined') lucide.createIcons(); 
    window.newConversation(); 
    window.loadHistory();
}

window.newConversation = () => {
    curChatId = null;
    const w = document.getElementById('chat-window');
    if(!w) return;
    
    if(currentMode === 'translate') {
        w.innerHTML = `
            <div id="empty-state" class="text-center py-12">
                <h1 class="text-3xl font-extrabold bg-gradient-to-r from-blue-500 to-red-500 bg-clip-text text-transparent">Llink Traduction</h1>
                <p class="text-gray-500 mt-2">Posez vos questions ou lancez une traduction.</p>
            </div>`;
    } else {
        w.innerHTML = `
            <div id="empty-state" class="text-center py-12">
                <h1 class="text-4xl font-extrabold bg-gradient-to-r from-blue-500 to-red-500 bg-clip-text text-transparent mb-4">Llink</h1>
                <p class="text-gray-500 mb-8">Posez vos questions ou lancez une discussion.</p>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
                    <button onclick="window.sendDirect('🎓 M\\\'aider à apprendre')" class="p-4 border border-gray-300 dark:border-gray-700 rounded-xl text-left hover:bg-gray-100 dark:hover:bg-gray-800 transition">
                        <span class="font-bold text-blue-500 block">🎓 M'aider à apprendre</span>
                        <span class="text-xs text-gray-500">Générer des explications méthodiques pour vos cours.</span>
                    </button>
                    <button onclick="window.sendDirect('✍️ M\\\'aider à rédiger')" class="p-4 border border-gray-300 dark:border-gray-700 rounded-xl text-left hover:bg-gray-100 dark:hover:bg-gray-800 transition">
                        <span class="font-bold text-green-500 block">✍️ M'aider à rédiger</span>
                        <span class="text-xs text-gray-500">Perfectionnez vos lettres, codes et documents.</span>
                    </button>
                </div>
            </div>`;
    }
};

window.loadHistory = async () => {
    if(!window.isLoggedIn) return;
    const h = await db.chats.where('mode').equals(currentMode).reverse().toArray();
    const historyDiv = document.getElementById('history');
    if(!historyDiv) return;

    if(h.length === 0) {
        historyDiv.innerHTML = `<div class="text-center p-4 text-sm font-bold text-gray-500 mt-10">Aucun historique</div>`;
        return;
    }

    historyDiv.innerHTML = h.map(c => 
        `<div class="flex items-center justify-between p-2 hover:bg-gray-200 dark:hover:bg-gray-800 rounded-lg cursor-pointer group">
            <div class="flex-grow truncate text-sm font-medium" onclick="window.switchChat(${c.id})">${c.title}</div>
            <div class="flex gap-2 opacity-0 group-hover:opacity-100 transition">
                <i data-lucide="edit-2" class="w-4 h-4 text-gray-500 hover:text-blue-500" onclick="window.openRenameModal(${c.id}, \`${c.title.replace(/`/g, "")}\`, event)"></i>
                <i data-lucide="trash-2" class="w-4 h-4 text-gray-500 hover:text-red-500" onclick="window.deleteChat(${c.id}, event)"></i>
            </div>
        </div>`
    ).join('');
    if(typeof lucide !== 'undefined') lucide.createIcons();
};

window.openRenameModal = (id, currentTitle, event) => {
    if(event) event.stopPropagation();
    chatToRename = id;
    const input = document.getElementById('rename-input');
    const modal = document.getElementById('rename-modal');
    if(input) input.value = currentTitle;
    if(modal) modal.style.display = 'flex';
};

window.saveChatName = async () => {
    const input = document.getElementById('rename-input');
    if (!input || !chatToRename) return;
    const newTitle = input.value.trim();
    if (newTitle) {
        await db.chats.update(chatToRename, { title: newTitle });
        window.loadHistory();
        window.closeModal('rename-modal');
        const fullChat = await db.chats.get(chatToRename);
        fullChat.messages = await db.msgs.where('chatId').equals(chatToRename).toArray();
        await syncChatToServer(fullChat);
    }
};

window.switchChat = async (id) => {
    curChatId = id;
    const m = await db.msgs.where('chatId').equals(id).toArray();
    const w = document.getElementById('chat-window');
    if(!w) return;
    w.innerHTML = "";
    m.forEach(msg => renderMessage(msg.text, msg.role));
    if(window.innerWidth < 1024) toggleSidebar();
};

window.deleteChat = async (id, event) => { 
    if(event) event.stopPropagation();
    if(confirm("Supprimer la discussion ?")) { 
        await db.chats.delete(id);
        await db.msgs.where('chatId').equals(id).delete();
        if(curChatId === id) window.newConversation(); 
        window.loadHistory(); 
    } 
};

function parseMarkdown(text) {
    if(!text) return "";
    let html = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    html = html.replace(/```([\s\S]*?)```/g, function(m, p1){ 
        return `<div class="bg-gray-900 text-gray-100 p-3 rounded-lg my-2 overflow-x-auto relative"><button onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText); this.innerText='Copié!'; setTimeout(()=>this.innerText='Copier',2000);" class="absolute right-2 top-2 bg-gray-800 text-xs px-2 py-1 rounded text-gray-400">Copier</button><pre><code>${p1}</code></pre></div>`; 
    });
    html = html.replace(/\*\*([\s\S]*?)\*\*/g, '<strong>$1</strong>'); 
    html = html.replace(/\n/g, '<br>'); 
    return html;
}

function renderMessage(txt, role, id="") {
    const w = document.getElementById('chat-window');
    if(!w) return;
    const html = parseMarkdown(txt);
    const safeTxt = encodeURIComponent(txt);
    
    const actionBtns = `
        <div class="flex gap-2 mt-1 justify-end opacity-60 hover:opacity-100 transition">
            <button onclick="navigator.clipboard.writeText(decodeURIComponent('${safeTxt}')); alert('Copié !')" class="p-1 hover:bg-gray-200 dark:hover:bg-gray-800 rounded"><i data-lucide="copy" class="w-3 h-3"></i></button>
        </div>
    `;

    if(role === 'user') {
        w.insertAdjacentHTML('beforeend', `
            <div class="flex flex-col items-end mb-4" id="${id}">
                <div class="bg-blue-600 text-white p-3 rounded-2xl rounded-tr-none max-w-[85%]">${html}</div>
                ${actionBtns}
            </div>`);
    } else {
        w.insertAdjacentHTML('beforeend', `
            <div class="flex flex-col items-start mb-4" id="${id}">
                <div class="flex items-center gap-2 mb-1 text-xs font-bold opacity-70">
                    <span class="text-red-500">LLINK</span>
                </div>
                <div class="bg-gray-200 dark:bg-gray-800 p-3 rounded-2xl rounded-tl-none max-w-[85%] msg-content">${html}</div>
                ${actionBtns}
            </div>`);
    }
    if(typeof lucide !== 'undefined') lucide.createIcons(); 
    scrollToBottom();
}

window.sendDirect = (txt) => { 
    const input = document.getElementById('user-msg');
    if(input) {
        input.value = txt; 
        window.sendFromInput(); 
    }
};

window.sendFromInput = async () => {
    const input = document.getElementById('user-msg');
    if(!input) return;
    const txt = input.value;
    if(!txt.trim()) return; 
    
    input.value = ''; 
    input.style.height = 'auto'; 
    
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
    
    const loadElem = document.getElementById(loadId);
    if(!loadElem) return;
    const contentDiv = loadElem.querySelector('.msg-content');
    let fullReply = "";

    if(window.isOnline) {
        try {
            const prefs = localStorage.getItem('llink_prefs') || "";
            const modelSelector = document.getElementById('model-selector');
            const selectedModel = modelSelector ? modelSelector.value : "gemini-1.5-flash";
            
            const payload = { message: txt, mode: currentMode, model: selectedModel, preferences: prefs };
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
        // --- PASSAGE MAÎTRE PAR LA LOGIQUE MODULE DU HTML ---
        if(!window.isOfflineValid) {
            fullReply = "❌ Votre abonnement pro a expiré. Mode hors-ligne indisponible.";
            contentDiv.innerHTML = fullReply;
        } else if(typeof window.generateOfflineText === "function") {
            fullReply = await window.generateOfflineText(txt, loadId);
            contentDiv.innerHTML = parseMarkdown(fullReply);
            scrollToBottom();
        } else {
            fullReply = "❌ Échec de liaison avec le module IA local.";
            contentDiv.innerHTML = fullReply;
        }
    }
    
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
    if(!f) return; 
    toggleMediaMenu();
    if(!window.isOnline) {
        renderMessage("OCR Local...", 'bot', 'ocr-load');
        if(typeof Tesseract !== 'undefined') {
            const {data:{text}} = await Tesseract.recognize(f, 'fra+eng');
            const loadOcr = document.getElementById('ocr-load');
            if(loadOcr) loadOcr.remove(); 
            window.sendDirect("Analyse cette image: " + text); 
        } else {
            alert("Moteur OCR non chargé.");
        }
        return;
    }
    const reader = new FileReader(); 
    reader.readAsDataURL(f);
    reader.onload = async () => {
        renderMessage("Analyse d'image Cloud...", 'bot', 'ocr-load');
        try {
            const res = await fetch(`${SERVER_URL}/api/ocr`, { method: 'POST', body: JSON.stringify({imageBase64: reader.result}), headers: {"Content-Type":"application/json"} });
            const data = await res.json(); 
            const loadOcr = document.getElementById('ocr-load');
            if(loadOcr) loadOcr.remove(); 
            window.sendDirect("Analyse: " + data.response);
        } catch(e) { 
            const loadOcr = document.getElementById('ocr-load');
            if(loadOcr) loadOcr.remove(); 
            alert("Erreur OCR Cloud"); 
        }
    }; 
    e.target.value = '';
};

window.handleVoice = async () => {
    if(!window.isOnline) { alert("Le micro est désactivé hors-ligne."); return; }
    const micBtn = document.getElementById('mic-btn');
    if(!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({audio:true});
            mediaRecorder = new MediaRecorder(stream); 
            audioChunks = [];
            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const blob = new Blob(audioChunks, {type:'audio/wav'});
                const form = new FormData(); 
                form.append("file", blob);
                try { 
                    const res = await fetch(`${SERVER_URL}/api/audio`, {method:'POST', body:form});
                    const data = await res.json(); 
                    if(data.text) window.sendDirect(data.text); 
                } catch(e){ 
                    alert("Erreur serveur audio."); 
                }
            };
            mediaRecorder.start(); 
            isRecording = true; 
            if(micBtn) micBtn.classList.add('text-red-500', 'animate-pulse');
        } catch(e) { 
            alert("Refus d'accès micro."); 
        }
    } else { 
        if(mediaRecorder) mediaRecorder.stop(); 
        isRecording = false; 
        if(micBtn) micBtn.classList.remove('text-red-500', 'animate-pulse'); 
    }
};

// Initialisation sécurisée des événements du DOM après chargement
document.addEventListener('DOMContentLoaded', () => {
    const fileIn = document.getElementById('file-in');
    const cameraIn = document.getElementById('camera-in');
    if(fileIn) fileIn.addEventListener('change', window.handleImageUpload);
    if(cameraIn) cameraIn.addEventListener('change', window.handleImageUpload);
    
    if(typeof lucide !== 'undefined') lucide.createIcons(); 
    window.newConversation();
});
