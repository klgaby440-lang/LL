const db = new Dexie("LlinkDB");
        db.version(2).stores({ chats: '++id, title, time, mode', msgs: '++id, chatId, role, text' });
        
        const SERVER_URL = "https://llink-usz9.onrender.com";
        let curChatId = null;
        window.isOnline = true; let currentMode = 'chat';
        window.isLoggedIn = false;
        window.isOfflineValid = false;
        let chatToRename = null;
        
        document.addEventListener('DOMContentLoaded', () => { document.getElementById('pref-input').value = localStorage.getItem('llink_prefs') || ""; });
        function savePreferences() { localStorage.setItem('llink_prefs', document.getElementById('pref-input').value); alert("Préférences sauvegardées !"); toggleSettingPanel('panel-pref');}

        window.checkSubscription = async () => {
            if(!window.isLoggedIn) return;
            try {
                const res = await fetch(`${SERVER_URL}/api/user/status`, { headers: {"Authorization": `Bearer ${window.authToken}`} });
                const data = await res.json();
                
                // GESTION DU MESSAGE DE BIENVENUE 3 JOURS
                if(data.isNewUser) alert("Bienvenue sur Llink ! Vous bénéficiez d'une période d'essai de 3 jours à partir d'aujourd'hui. Il vous faudra vous abonner après cela.");
                
                window.isOfflineValid = data.isValid;
                const infoDiv = document.getElementById('subscription-info');
                
                if(data.isValid && data.isTrial) infoDiv.innerHTML = `🟢 Période d'essai valide jusqu'au : ${new Date(data.expirationDate).toLocaleDateString()}`;
                else if(data.isValid && !data.isTrial) infoDiv.innerHTML = `💎 Abonnement Pro valide jusqu'au : ${new Date(data.expirationDate).toLocaleDateString()}`;
                else { infoDiv.innerHTML = `🔴 Accès Hors-Ligne expiré.`; window.isOfflineValid = false; }
                
                updateOfflineUI();
            } catch(e) { console.error("checkSubscription error", e); }
        };

        async function submitSubscription() {
            if(!window.isLoggedIn) { alert("Vous devez être connecté pour vous abonner."); return; }
            const txId = document.getElementById('transaction-id').value.trim();
            if(!txId) return;
            document.getElementById('subscription-info').innerText = "Vérification en cours...";
            try {
                const res = await fetch(`${SERVER_URL}/api/verify-payment`, {
                    method: 'POST', headers: {"Content-Type":"application/json", "Authorization": `Bearer ${window.authToken}`},
                    body: JSON.stringify({ transactionId: txId })
                });
                const data = await res.json();
                if(data.success) { alert("Paiement validé avec succès !"); await window.checkSubscription(); document.getElementById('transaction-id').value = ''; }
                else alert("Erreur : ID de transaction invalide.");
            } catch(e) { alert("Erreur réseau de validation."); }
        }

        async function syncChatToServer(chat) {
            if(!window.isLoggedIn) return;
            try { await fetch(`${SERVER_URL}/api/history/sync`, { method: "POST", headers: {"Content-Type":"application/json", "Authorization": `Bearer ${window.authToken}`}, body: JSON.stringify(chat) }); } catch(e){}
        }

        window.restoreHistory = async () => {
            try {
                const user = window.auth ? window.auth.currentUser : null;
                if (!user) return;
                const userToken = await user.getIdToken();
                const response = await fetch(`${SERVER_URL}/api/history`, { method: 'GET', headers: { 'Authorization': `Bearer ${userToken}`, 'Content-Type': 'application/json' } });
                if (!response.ok) return;
                const data = await response.json();
                if (!data.warning) console.log("Historique distant synchronisé.");
            } catch (error) { console.error("Erreur historique:", error); }
        };

        function updateOfflineUI() {
            const alertBox = document.getElementById('pro-alert');
            if(!window.isOnline && !window.isOfflineValid) alertBox.classList.remove('hidden');
            else alertBox.classList.add('hidden');
        }

        window.toggleOnline = () => {
            window.isOnline = !window.isOnline;
            const dot = document.getElementById('network-status');
            const modelSel = document.getElementById('model-selector');
            const micBtn = document.getElementById('mic-btn');
            if(window.isOnline) {
                dot.className = "w-[0.875rem] h-[0.875rem] rounded-full bg-green-500 shadow-[0_0_8px_#34A853]";
                modelSel.disabled = false; modelSel.style.opacity = 1;
                micBtn.disabled = false; micBtn.style.opacity = 1;
            } else {
                dot.className = "w-[0.875rem] h-[0.875rem] rounded-full bg-red-500";
                modelSel.disabled = true; modelSel.style.opacity = 0.5;
                micBtn.disabled = true; micBtn.style.opacity = 0.5;
                if(!window.isLoggedIn) alert("Connectez-vous pour utiliser le mode Hors-Ligne.");
            }
            updateOfflineUI();
        };

        function toggleTheme() { document.documentElement.classList.toggle('dark'); }
        function openSettings() { document.getElementById('settings-modal').style.display = 'flex'; if(window.innerWidth<768) toggleSidebar(); }
        function closeSettings() { document.getElementById('settings-modal').style.display = 'none'; }
        function toggleSettingPanel(id) { const p = document.getElementById(id); p.style.display = p.style.display === 'block' ? 'none' : 'block'; }
        function toggleSidebar() { document.getElementById('sidebar').classList.toggle('-translate-x-full'); }
        function toggleMediaMenu() { const m = document.getElementById('media-menu'); m.style.display = m.style.display==='flex'?'none':'flex'; }
        function triggerInput(id) { document.getElementById(id).click(); toggleMediaMenu(); }
        function scrollToBottom() { const w = document.getElementById('chat-window'); w.scrollTop = w.scrollHeight; }
        window.closeModal = (id) => { document.getElementById(id).style.display = 'none'; };
        
        async function toggleMode() {
            currentMode = currentMode === 'chat' ? 'translate' : 'chat';
            const btn = document.getElementById('mode-btn'); const lang = document.getElementById('lang-bar');
            if(currentMode === 'chat') { btn.innerHTML = '<i data-lucide="message-square" class="w-[1.25rem] h-[1.25rem]"></i>'; lang.classList.add('hidden'); lang.classList.remove('flex'); }
            else { btn.innerHTML = '<i data-lucide="languages" class="w-[1.25rem] h-[1.25rem] text-blue-500"></i>'; lang.classList.remove('hidden'); lang.classList.add('flex'); }
            lucide.createIcons(); await newConversation(); window.loadHistory();
        }

        window.newConversation = () => {
            curChatId = null;
            const w = document.getElementById('chat-window');
            if(currentMode === 'translate') {
                w.innerHTML = `<div id="empty-state" class="flex flex-col items-center justify-center h-full text-center mt-[2.5rem]"><div class="logo-anim opacity-20 dark:opacity-10 mb-[1.5rem] flex justify-center"><svg class="logo-llink overflow-visible" viewBox="-5 -5 110 110"><circle cx="50" cy="20" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="76" cy="35" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="76" cy="65" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="50" cy="80" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="24" cy="65" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="24" cy="35" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/></svg></div><h1 class="text-[1.875rem] font-extrabold bg-gradient-to-r from-blue-500 to-red-500 bg-clip-text text-transparent">Llink Traduction</h1><p class="text-[0.875rem] font-bold text-gray-500 mt-[0.5rem]">Posez vos questions ou lancez une traduction.</p></div>`;
            } else {
                w.innerHTML = `<div class="flex flex-col items-center justify-center h-full text-center mt-10"><div class="logo-anim opacity-20 dark:opacity-10 mb-4"><svg width="100" height="100" viewBox="-20 0 100 100"><circle cx="50" cy="20" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="76" cy="35" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="76" cy="65" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="50" cy="80" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="24" cy="65" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="24" cy="35" r="18" fill="none" stroke="currentColor" stroke-width="2.5"/></svg></div><h1 class="text-3xl font-extrabold bg-gradient-to-r from-blue-500 to-red-500 bg-clip-text text-transparent">Llink</h1><p class="text-sm font-bold text-gray-500 mt-1 mb-6">Posez vos questions ou lancez une traduction.</p>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl text-left">
                    <button onclick="sendDirect('M\\'aider à apprendre')" class="p-4 border border-gray-300 dark:border-gray-700 rounded-2xl hover:bg-surface-light dark:hover:bg-surface-dark transition flex flex-col gap-1"><span class="font-black text-blue-400">🎓 M'aider à apprendre</span><span class="text-xs text-gray-500 font-bold">générer des explications méthodiques pour vos cours avec des exercices adaptés.</span></button>
                    <button onclick="sendDirect('M\\'aider à rédiger')" class="p-4 border border-gray-300 dark:border-gray-700 rounded-2xl hover:bg-surface-light dark:hover:bg-surface-dark transition flex flex-col gap-1"><span class="font-black text-green-500">✍️ M'aider à rédiger</span><span class="text-xs text-gray-500 font-bold">perfectionnez vos lettre, codes sources et autres documents écris.</span></button>
                    <button onclick="sendDirect('Discuter de quelque chose')" class="p-4 border border-gray-300 dark:border-gray-700 rounded-2xl hover:bg-surface-light dark:hover:bg-surface-dark transition flex flex-col gap-1"><span class="font-black text-purple-500">💬 Discuter de quelque chose</span><span class="text-xs text-gray-500 font-bold">explorer des thématiques des sujets qui vous passionent.</span></button>
                    <button onclick="sendDirect('Me surprendre')" class="p-4 border border-gray-300 dark:border-gray-700 rounded-2xl hover:bg-surface-light dark:hover:bg-surface-dark transition flex flex-col gap-1"><span class="font-black text-orange-500">✨ Me surprendre</span><span class="text-xs text-gray-500 font-bold">laissez Llink vous suprendre avec quelque chose d'inatendue.</span></button>
                    <button onclick="sendDirect('Comment utiliser Llink ?')" class="p-4 border border-gray-300 dark:border-gray-700 rounded-2xl hover:bg-surface-light dark:hover:bg-surface-dark transition flex flex-col gap-1 md:col-span-2 text-center items-center"><span class="font-black text-blue-600">🚀 Comment utiliser</span><span class="text-xs text-gray-500 font-bold">comprendre en quoi Llink different des IA actuelles et comment l'utiliser.</span></button>
                </div></div>`;
            }
        };

        window.loadHistory = async () => {
            if(!window.isLoggedIn) return;
            const h = await db.chats.where('mode').equals(currentMode).reverse().toArray();
            document.getElementById('history').innerHTML = h.map(c => 
                `<div class="flex items-center justify-between p-[0.75rem] rounded-xl hover:bg-gray-300 dark:hover:bg-gray-800 cursor-pointer font-bold text-[0.875rem] transition group">
                    <div class="flex-grow truncate pr-2" onclick="switchChat(${c.id})">${c.title}</div>
                    <div class="flex items-center gap-[0.75rem] opacity-100 lg:opacity-0 lg:group-hover:opacity-100 transition-opacity">
                        <i data-lucide="edit-2" class="w-[1rem] h-[1rem] text-blue-400 hover:text-blue-600" onclick="openRenameModal(${c.id}, \`${c.title.replace(/`/g, "")}\`, event)"></i>
                        <i data-lucide="trash-2" class="w-[1rem] h-[1rem] text-red-400 hover:text-red-600" onclick="deleteChat(${c.id}, event)"></i>
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
            if(window.innerWidth<768) toggleSidebar();
        };

        window.deleteChat = async (id, event) => { 
            if(event) event.stopPropagation();
            if(confirm("Supprimer la discussion ?")) { 
                await db.chats.delete(id);
                await db.msgs.where('chatId').equals(id).delete();
                if(curChatId===id) newConversation(); 
                window.loadHistory(); 
            } 
        };

        function parseMarkdown(text) {
            if(!text) return "";
            let html = text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            html = html.replace(/```([\s\S]*?)```/g, function(m, p1){ return `<div class="relative"><button onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText); this.innerText='Copié!'; setTimeout(()=>this.innerText='Copier',2000);" class="absolute top-[0.5rem] right-[0.5rem] bg-gray-700 text-white text-[0.625rem] font-bold py-[0.25rem] px-[0.5rem] rounded">Copier</button><pre><code>${p1}</code></pre></div>`; });
            html = html.replace(/\*\*([\s\S]*?)\*\*/g, '<strong>$1</strong>'); html = html.replace(/\n/g, '<br>'); return html;
        }

        // ================= INJECTION DES BOUTONS COPIER / PARTAGER =================
        function renderMessage(txt, role, id="") {
            const w = document.getElementById('chat-window');
            const html = parseMarkdown(txt);
            
            // On encode le texte pour éviter que des guillemets cassent le code HTML du bouton
            const safeTxt = encodeURIComponent(txt);
            
            const actionBtns = `
                <div class="msg-actions flex gap-[0.75rem] mt-[0.5rem] px-[0.5rem] text-gray-500">
                    <button onclick="navigator.clipboard.writeText(decodeURIComponent('${safeTxt}')); this.innerHTML='<i data-lucide=\\'check\\' class=\\'w-[1rem] h-[1rem] text-green-500\\'></i>'; setTimeout(() => { this.innerHTML='<i data-lucide=\\'copy\\' class=\\'w-[1rem] h-[1rem]\\'></i>'; lucide.createIcons(); }, 2000);" title="Copier" class="hover:text-blue-500 transition flex items-center justify-center p-[0.25rem]"><i data-lucide="copy" class="w-[1rem] h-[1rem]"></i></button>
                    <button onclick="if(navigator.share) { navigator.share({title: 'Llink', text: decodeURIComponent('${safeTxt}')}) } else { alert('Partage non supporté.'); }" title="Partager" class="hover:text-green-500 transition flex items-center justify-center p-[0.25rem]"><i data-lucide="share-2" class="w-[1rem] h-[1rem]"></i></button>
                </div>
            `;

            if(role === 'user') {
                w.insertAdjacentHTML('beforeend', `<div class="flex flex-col items-end w-full msg-wrapper" id="${id}"><div class="bg-userBg-light dark:bg-userBg-dark p-[1rem] rounded-3xl rounded-br-sm max-w-[85%] font-medium msg-content text-[1rem] shadow-sm">${html}</div>${actionBtns}</div>`);
            } else {
                w.insertAdjacentHTML('beforeend', `<div class="flex flex-col items-start w-full msg-wrapper" id="${id}"><div class="flex items-center gap-[0.5rem] mb-[0.25rem]"><div class="w-[2rem] h-[2rem] rounded-full logo-anim flex items-center justify-center flex-shrink-0 bg-gray-200 dark:bg-gray-800"><svg viewBox="-5 -5 110 110" class="logo-llink overflow-visible w-full h-full"><circle cx="50" cy="20" r="18" fill="none" stroke="#FF5733" stroke-width="8"/><circle cx="76" cy="35" r="18" fill="none" stroke="#33FF57" stroke-width="8"/><circle cx="76" cy="65" r="18" fill="none" stroke="#3357FF" stroke-width="8"/><circle cx="50" cy="80" r="18" fill="none" stroke="#F333FF" stroke-width="8"/><circle cx="24" cy="65" r="18" fill="none" stroke="#FFBD33" stroke-width="8"/><circle cx="24" cy="35" r="18" fill="none" stroke="#33FFF3" stroke-width="8"/></svg></div><span class="text-[0.75rem] font-black uppercase tracking-wider">LLINK</span></div><div class="px-[0.5rem] w-full font-medium msg-content text-[1rem]">${html}</div>${actionBtns}</div>`);
            }
            lucide.createIcons(); scrollToBottom();
        }

        window.sendDirect = (txt) => { document.getElementById('user-msg').value = txt; sendFromInput(); };
        
        window.sendFromInput = async () => {
            const input = document.getElementById('user-msg');
            const txt = input.value;
            if(!txt.trim()) return; input.value = ''; input.style.height = '3rem';
            if(!curChatId) { curChatId = await db.chats.add({title: txt.substring(0,20), time: Date.now(), mode: currentMode}); window.loadHistory(); }
            
            const emptyState = document.getElementById('empty-state');
            if(emptyState) emptyState.remove();
            
            if(window.isLoggedIn) await db.msgs.add({chatId: curChatId, role: 'user', text: txt});
            renderMessage(txt, 'user');
            
            const loadId = "load-"+Date.now(); renderMessage("Réflexion...", 'bot', loadId);
            const contentDiv = document.getElementById(loadId).querySelector('.msg-content');
            let fullReply = "";

            if(window.isOnline) {
                try {
                    const prefs = localStorage.getItem('llink_prefs') || "";
                    const payload = { message: txt, mode: currentMode, model: document.getElementById('model-selector').value, preferences: prefs };
                    const res = await fetch(`${SERVER_URL}/api/chat`, { method: 'POST', body: JSON.stringify(payload), headers: {"Content-Type":"application/json", "Authorization": window.authToken?`Bearer ${window.authToken}`:""} });
                    const reader = res.body.getReader();
                    const decoder = new TextDecoder(); contentDiv.innerHTML = "";
                    while(true) { const {value, done} = await reader.read(); if(done) break;
                    fullReply += decoder.decode(value, {stream:true}); contentDiv.innerHTML = parseMarkdown(fullReply); scrollToBottom(); }
                } catch(e) { fullReply = "❌ Erreur de connexion au serveur Render."; contentDiv.innerHTML = fullReply; }
            } else {
                if(!window.isOfflineValid) { fullReply = "❌ Votre accès au mode Hors-Ligne est expiré. Veuillez vous connecter à Internet et vous abonner."; }
                else if(window.generateOfflineText) { fullReply = await window.generateOfflineText(txt, loadId); }
                else { fullReply = "⚠️ Modèle WASM non prêt."; }
                contentDiv.innerHTML = parseMarkdown(fullReply);
            }

            if(window.isLoggedIn) {
                await db.msgs.add({chatId: curChatId, role: 'bot', text: fullReply});
                const fullChat = await db.chats.get(curChatId); fullChat.messages = await db.msgs.where('chatId').equals(curChatId).toArray();
                await syncChatToServer(fullChat);
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

        lucide.createIcons(); newConversation();
