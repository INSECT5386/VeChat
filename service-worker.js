// 루트 또는 public/ 아래에 service-worker.js
const CACHE_NAME = 'covec-cache-v58';
const urlsToCache = [
  '/VeChat/',
  '/VeChat/index.html',
  '/VeChat/css/style.css',
  '/VeChat/js/script.js',
  '/VeChat/manifest.json',
  '/VeChat/img/icon-192.png',
  '/VeChat/img/icon-512.png',
  'https://cdn.jsdelivr.net/npm/marked/marked.min.js',
  'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/highlight.min.js',
  'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/styles/atom-one-light.min.css'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('📦 캐시 저장 시작');
      return cache.addAll(urlsToCache);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then(response => {
      // 캐시에 있으면 반환, 없으면 네트워크 요청
      return response || fetch(event.request);
    })
  );
});
