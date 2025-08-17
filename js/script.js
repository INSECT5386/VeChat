// === 외부 라이브러리 동적 로드 ===
(function loadLibraries() {
  const markedScript = document.createElement('script');
  markedScript.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
  markedScript.onload = () => {
    window.marked = marked;
    loadHighlightJS();
  };
  document.head.appendChild(markedScript);

  function loadHighlightJS() {
    const hljsScript = document.createElement('script');
    hljsScript.src = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/highlight.min.js';
    hljsScript.onload = () => {
      window.hljs = hljs;
      document.addEventListener('DOMContentLoaded', () => {
        hljs.configure({ ignoreIllegals: true });
      });
    };
    document.head.appendChild(hljsScript);
  }
})();

// === 상태 관리 ===
const state = { rooms: {}, currentRoom: null };

// === 세션 ID 관리 ===
let sessionId = localStorage.getItem('session_id') || crypto.randomUUID();
localStorage.setItem('session_id', sessionId);

// === DOM 요소 ===
const $ = (sel) => document.querySelector(sel);
const roomListUl = $('#room-list ul');
const inputText = $('#input-text');
const sendBtn = $('#send-btn');
const chatContainer = $('#chat-container');

// === 초기화 ===
function init() {
  createRoom('기본 채팅방');
  switchRoom('기본 채팅방');
  autoResize();

  const sample = `안녕하세요!`;
  addMessage(sample, false);
}

// === 채팅방 생성/전환 ===
function createRoom(name) {
  if (state.rooms[name]) { alert('이미 존재하는 방입니다!'); return false; }
  state.rooms[name] = [];
  renderRoomList();
  return true;
}

function switchRoom(name) {
  if (!state.rooms[name]) return;
  state.currentRoom = name;
  renderRoomList();
  renderMessages();
}

// === 렌더링 ===
function renderRoomList() {
  roomListUl.innerHTML = '';
  Object.keys(state.rooms).forEach(roomName => {
    const li = document.createElement('li');
    li.textContent = roomName;
    if (roomName === state.currentRoom) li.classList.add('active');
    li.addEventListener('click', () => switchRoom(roomName));
    roomListUl.appendChild(li);
  });
}

function renderMessages() {
  chatContainer.innerHTML = '';
  const messages = state.rooms[state.currentRoom] || [];
  messages.forEach(msg => addMessageToUI(msg.text, msg.fromUser, false));
}

// === 메시지 추가 ===
function addMessage(text, fromUser, save = true) {
  addMessageToUI(text, fromUser, true);
  if (save) state.rooms[state.currentRoom].push({ text, fromUser });
}

function addMessageToUI(text, fromUser, animate = true) {
  const div = document.createElement('div');
  div.classList.add('message', fromUser ? 'user' : 'bot');
  if (!animate) div.style.animation = 'none';
  div.innerHTML = marked.parse(text);

  div.querySelectorAll('pre code').forEach(block => {
    block.classList.add('hljs');
    hljs.highlightElement(block);
    makeCopyButton(block);
  });

  chatContainer.appendChild(div);
  scrollToBottom();
}

// === 복사 버튼 ===
function makeCopyButton(codeBlock) {
  if (codeBlock.parentNode.querySelector('.copy-btn')) return;
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.textContent = '복사';
  btn.onclick = () => {
    navigator.clipboard.writeText(codeBlock.innerText)
      .then(() => { btn.textContent = '복사됨!'; setTimeout(() => btn.textContent = '복사', 1500); })
      .catch(() => { btn.textContent = '실패'; setTimeout(() => btn.textContent = '복사', 1500); });
  };
  codeBlock.parentNode.style.position = 'relative';
  codeBlock.parentNode.appendChild(btn);
}

// === 유틸 ===
function scrollToBottom() { chatContainer.scrollTop = chatContainer.scrollHeight; }
function autoResize() { inputText.style.height = 'auto'; inputText.style.height = inputText.scrollHeight + 'px'; }

// === 메시지 전송 ===
async function sendMessage() {
  const text = inputText.value.trim();
  if (!text) return;

  addMessage(text, true);
  inputText.value = '';
  autoResize();
  sendBtn.disabled = true;

  const botDiv = document.createElement('div');
  botDiv.classList.add('message', 'bot');
  chatContainer.appendChild(botDiv);
  scrollToBottom();

  let fullText = '';
  try {
    const encodedMsg = encodeURIComponent(text);
    const url = `https://yuchan5386-vechat.hf.space/api/chat?message=${encodedMsg}&session_id=${sessionId}`;
    const res = await fetch(url, { method: 'GET' });

    if (!res.ok) throw new Error('서버 오류');

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      chunk.split('\n').forEach(line => {
        if (!line.startsWith('data:')) return;
        const dataStr = line.slice(5).trim();
        if (!dataStr) return;

        try {
          const data = JSON.parse(dataStr);
          if (data.char !== undefined) {
            fullText += data.char;
            botDiv.innerHTML = marked.parse(fullText);
            botDiv.querySelectorAll('pre code').forEach(block => {
              hljs.highlightElement(block);
              makeCopyButton(block);
            });
            scrollToBottom();
          } else if (data.done) {
            botDiv.innerHTML = marked.parse(fullText);
            botDiv.querySelectorAll('pre code').forEach(block => {
              hljs.highlightElement(block);
              makeCopyButton(block);
            });
            scrollToBottom();
          } else if (data.error) {
            botDiv.textContent = '⚠️ ' + data.error;
          }
        } catch (e) {
          console.warn('JSON Parse Error:', e);
        }
      });
    }
  } catch (e) {
    botDiv.textContent = '⚠️ 통신 실패: ' + e.message;
  } finally {
    sendBtn.disabled = false;
    autoResize();
    scrollToBottom();
  }
}

// === 이벤트 바인딩 ===
inputText.addEventListener('input', () => { autoResize(); sendBtn.disabled = !inputText.value.trim(); });
inputText.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
sendBtn.addEventListener('click', sendMessage);
$('#new-room-btn').addEventListener('click', () => {
  const name = prompt('새 채팅방 이름은?');
  if (name && createRoom(name)) switchRoom(name);
});

// === 초기 실행 ===
init();
