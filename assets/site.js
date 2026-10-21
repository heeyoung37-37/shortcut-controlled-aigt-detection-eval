'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const domainNames = {essay:'IELTS-style essays',arxiv_abstract:'Research abstracts',blog:'Blog passages',product_review:'Consumer product reviews',peer_review:'Scientific peer reviews',meeting_summary:'Municipal action summaries',translation:'Translation into English'};
  const variantNames = {raw_ai:'AI generation',ai_edited:'AI-edited text',human_edited:'Human-edited AI text'};
  const state = {domain:'essay',model:'gpt-5.5',variant:'raw_ai',index:0,blind:false};
  const cache = new Map();
  let pairs = [], sampleRequest = 0, promptRequest = 0, promptText = '', toastTimer;
  function toast(message) {
    $('toast').textContent = message; $('toast').classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('show'), 2600);
  }
  async function read(path) {
    if (window.AIGT_PREVIEW_FILES && Object.hasOwn(window.AIGT_PREVIEW_FILES,path)) return window.AIGT_PREVIEW_FILES[path];
    const response = await fetch(path);
    if (!response.ok) throw new Error('Could not load ' + path);
    return response.text();
  }
  function loadMessage() {
    return location.protocol === 'file:' ? 'Open this folder through a local web server to load the data: python3 -m http.server 8765 --bind 127.0.0.1. Then visit http://127.0.0.1:8765/.' : 'This file could not be loaded. Check that the data and prompts folders are next to index.html, then reload the page.';
  }
  function markSelected(selector, key, value) {
    document.querySelectorAll(selector).forEach(button => button.setAttribute('aria-pressed', String(button.dataset[key] === value)));
  }
  async function loadPairs() {
    const request = ++sampleRequest;
    const {domain,model,variant} = state;
    $('text-pair').setAttribute('aria-busy','true'); $('load-error').hidden = true;
    ['prev','next','download-pair'].forEach(id => $(id).disabled = true);
    $('human-text').textContent = 'Loading the original text…'; $('ai-text').textContent = 'Loading the paired text…';
    $('human-count').textContent = ''; $('ai-count').textContent = ''; $('sample-counter').textContent = '— / —';
    $('domain-caption').textContent = domainNames[domain];
    try {
      if (!cache.has(domain)) cache.set(domain, read('data/' + domain + '.jsonl').then(text => text.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line))).catch(error => {cache.delete(domain); throw error;}));
      const all = await cache.get(domain);
      if (request !== sampleRequest) return;
      pairs = all.filter(row => row.model_label === model && row.variant_type === variant);
      if (!pairs.length) throw new Error('No pairs in this selection');
      state.index = 0; renderPair();
    } catch (error) {
      if (request !== sampleRequest) return;
      pairs = [];
      $('human-text').textContent = 'The benchmark is available in the data folder.';
      $('ai-text').textContent = 'Each JSONL row contains the original human reference and its AI-side pair.';
      $('load-error').textContent = loadMessage(); $('load-error').hidden = false;
    } finally {if(request === sampleRequest) $('text-pair').setAttribute('aria-busy','false');}
  }
  function renderPair() {
    const row = pairs[state.index];
    $('human-text').textContent = row.original; $('ai-text').textContent = row.sampled;
    $('human-count').textContent = row.original.trim().split(/\s+/u).length + ' words';
    $('ai-count').textContent = row.sampled.trim().split(/\s+/u).length + ' words';
    $('human-label').textContent = state.blind ? 'Text A' : 'Human reference';
    $('ai-label').textContent = state.blind ? 'Text B' : variantNames[state.variant];
    $('human-text').setAttribute('aria-label',state.blind ? 'Text A' : 'Human reference text');
    $('ai-text').setAttribute('aria-label',state.blind ? 'Text B' : 'AI-side text');
    $('sample-counter').textContent = String(state.index + 1).padStart(2,'0') + ' / ' + pairs.length;
    $('prev').disabled = state.index === 0; $('next').disabled = state.index === pairs.length-1;
    $('download-pair').disabled = false;
    $('human-text').scrollTop = 0; $('ai-text').scrollTop = 0;
  }
  document.querySelectorAll('[data-domain]').forEach(button => button.addEventListener('click',() => {
    state.domain = button.dataset.domain; markSelected('[data-domain]','domain',state.domain); loadPairs();
  }));
  $('model').addEventListener('change',event => {state.model = event.target.value; loadPairs();});
  document.querySelectorAll('[data-variant]').forEach(button => button.addEventListener('click',() => {
    state.variant = button.dataset.variant; markSelected('[data-variant]','variant',state.variant); loadPairs();
  }));
  $('prev').addEventListener('click',() => {if(state.index > 0) {state.index--; renderPair();}});
  $('next').addEventListener('click',() => {if(state.index < pairs.length-1) {state.index++; renderPair();}});
  $('blind-toggle').addEventListener('click',() => {
    state.blind = !state.blind;
    $('blind-toggle').textContent = state.blind ? 'Show authorship ◉' : 'Hide authorship ◉';
    $('blind-toggle').setAttribute('aria-pressed',String(state.blind));
    $('text-pair').classList.toggle('blind',state.blind);
    if(pairs.length) renderPair();
  });
  $('expand').addEventListener('click',() => {
    const expanded = $('text-pair').classList.toggle('expanded');
    $('expand').setAttribute('aria-expanded',String(expanded));
    $('expand').textContent = expanded ? 'Compact view ↙' : 'Expand text ↗';
  });
  $('download-pair').addEventListener('click',() => {
    const row = pairs[state.index]; if(!row) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(row,null,2)+'\n'],{type:'application/json'}));
    const link = document.createElement('a'); link.href = url;
    link.download = [state.domain,state.model,state.variant,String(state.index+1).padStart(2,'0')].join('_')+'.json';
    link.click(); setTimeout(() => URL.revokeObjectURL(url),1000); toast('Selected pair saved as JSON.');
  });
  // A small, non-HTML Markdown renderer. Dataset and prompt text never enter innerHTML.
  function inline(parent, text, basePath) {
    const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g;
    let last = 0;
    for (const match of text.matchAll(pattern)) {
      parent.append(document.createTextNode(text.slice(last,match.index)));
      const token = match[0]; let element;
      if(token.startsWith('`')) {element=document.createElement('code'); element.textContent=token.slice(1,-1);}
      else if(token.startsWith('**')) {element=document.createElement('strong'); element.textContent=token.slice(2,-2);}
      else {
        const link = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        let url;
        try {url = new URL(link[2],new URL(basePath,location.href));} catch {url=null;}
        if(url && ['http:','https:','file:'].includes(url.protocol)) {element=document.createElement('a'); element.textContent=link[1]; element.href=url.href; if(url.origin!==location.origin){element.rel='noreferrer';element.target='_blank';}}
        else {element=document.createElement('span');element.textContent=link[1];}
      }
      parent.append(element);last=match.index+token.length;
    }
    parent.append(document.createTextNode(text.slice(last)));
  }
  function markdown(target,text,basePath) {
    target.replaceChildren(); let code = null, list = null, table = null;
    for(const line of text.split(/\r?\n/)) {
      if(line.startsWith('```')) {list=null;if(code){code=null;} else {code=document.createElement('pre');target.append(code);}continue;}
      if(code){code.append(document.createTextNode(line+'\n'));continue;}
      if(!line.trim()){list=null;table=null;continue;}
      if(line.startsWith('|') && line.endsWith('|')) {
        const cells=line.slice(1,-1).split('|').map(cell=>cell.trim());
        if(cells.every(cell=>/^:?-+:?$/.test(cell))) continue;
        const header = !table;
        if(!table){const scroll=document.createElement('div');scroll.className='table-scroll';table=document.createElement('table');scroll.append(table);target.append(scroll);}
        const row=document.createElement('tr');
        cells.forEach(text=>{const cell=document.createElement(header?'th':'td');inline(cell,text,basePath);row.append(cell);});table.append(row);continue;
      }
      table=null;
      const heading=line.match(/^(#{1,3})\s+(.+)$/);
      if(heading){list=null;const node=document.createElement('h'+heading[1].length);inline(node,heading[2],basePath);target.append(node);continue;}
      if(line.startsWith('- ')){if(!list){list=document.createElement('ul');target.append(list);}const item=document.createElement('li');inline(item,line.slice(2),basePath);list.append(item);continue;}
      list=null; const paragraph=document.createElement('p');inline(paragraph,line,basePath);target.append(paragraph);
    }
  }
  async function loadPrompt(name) {
    const request = ++promptRequest;
    markSelected('[data-prompt]','prompt',name); $('prompt-filename').textContent = name+'.md';
    $('prompt-document').setAttribute('aria-busy','true'); $('copy-prompt').disabled=true;
    $('prompt-document').textContent='Loading the construction template…';
    $('prompt-download').href='prompts/'+name+'.md';
    try {
      const text = await read('prompts/'+name+'.md'); if(request!==promptRequest)return;
      promptText=text; markdown($('prompt-document'),text,'prompts/'+name+'.md');
      $('prompt-document').scrollTop=0; $('copy-prompt').disabled=false;
    } catch {if(request===promptRequest) $('prompt-document').textContent=loadMessage();}
    finally {if(request===promptRequest) $('prompt-document').setAttribute('aria-busy','false');}
  }
  document.querySelectorAll('[data-prompt]').forEach(button=>button.addEventListener('click',()=>loadPrompt(button.dataset.prompt)));
  $('copy-prompt').addEventListener('click',async()=>{
    try {await navigator.clipboard.writeText(promptText);toast('Original Markdown copied.');}
    catch {toast('Copy is unavailable in this browser. Use Download original Markdown.');}
  });
  $('download-domain').addEventListener('change',event=>{$('domain-download').href='data/'+event.target.value+'.jsonl';});
  let documentRequest=0;
  document.querySelectorAll('[data-doc]').forEach(button=>button.addEventListener('click',async()=>{
    const request=++documentRequest; const path=button.dataset.doc;
    $('document-title').textContent=button.dataset.title; $('document-body').textContent='Loading…';
    $('document-dialog').showModal();
    try {const text=await read(path);if(request!==documentRequest)return;
      if(path.endsWith('.cff')) {const pre=document.createElement('pre');pre.textContent=text;$('document-body').replaceChildren(pre);}
      else markdown($('document-body'),text,path);
    } catch {if(request===documentRequest) $('document-body').textContent=loadMessage();}
  }));
  $('close-dialog').addEventListener('click',()=>$('document-dialog').close());
  $('document-dialog').addEventListener('click',event=>{if(event.target===$('document-dialog')){const box=event.target.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)event.target.close();}});
  loadPairs(); loadPrompt('generation');
})();
