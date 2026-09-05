import {spawn} from 'node:child_process';
import {mkdtempSync,writeFileSync,existsSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const root=resolve(process.argv[2]||'.');
const file=join(root,'mamba3_interactive_guide.html');
const out=process.argv[3]||tmpdir();
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless','--no-first-run','--no-default-browser-check','--remote-debugging-pipe',`--user-data-dir=${mkdtempSync(join(tmpdir(),'mamba-guide-'))}`],{stdio:['ignore','ignore','ignore','pipe','pipe']});
let id=0,buffer='';const pending=new Map(),errors=[];
chrome.stdio[4].on('data',chunk=>{buffer+=chunk.toString();let end;while((end=buffer.indexOf('\0'))>=0){const m=JSON.parse(buffer.slice(0,end));buffer=buffer.slice(end+1);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params);}});
function call(method,params={},sessionId){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});chrome.stdio[3].write(JSON.stringify({id:n,method,params,sessionId})+'\0');});}
const timeout=setTimeout(()=>{chrome.kill();process.exit(1)},45000);
try{
const {targetId}=await call('Target.createTarget',{url:'about:blank'});
const {sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
const c=(method,params)=>call(method,params,sessionId);
const evaluate=async expression=>{const r=await c('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});assert(!r.exceptionDetails,JSON.stringify(r.exceptionDetails));return r.result.value;};
await c('Runtime.enable');await c('Page.enable');await c('Network.enable');await c('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:0,uploadThroughput:0});
await c('Page.navigate',{url:pathToFileURL(file).href});
for(let i=0;i<100;i++){if(await evaluate("document.readyState==='complete' && !!document.getElementById('complex-svg')"))break;await new Promise(r=>setTimeout(r,50));}
const checks=[];
for(const [width,height] of [[1440,900],[390,844]]){
await c('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
const m=await evaluate('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})');assert(m.scrollWidth<=width,JSON.stringify(m));checks.push(m);
const shot=await c('Page.captureScreenshot',{format:'png'});writeFileSync(join(out,`guide-${width}.png`),Buffer.from(shot.data,'base64'));
}
const anchors=await evaluate("[...document.querySelectorAll('a')].map(a=>({href:a.href,valid:!a.getAttribute('href').startsWith('#')||!!document.querySelector(a.getAttribute('href'))}))");
for(const a of anchors){assert(a.valid,a.href);if(a.href.startsWith('file:')&&!a.href.endsWith('mamba3_receipts.json'))assert(existsSync(fileURLToPath(a.href.split('#')[0])),a.href);}
const cases=await evaluate(`(()=>{const results=[];for(const [a,w,d] of [[-.3,.8,.4],[0,3,1],[-2,.1,.1],[-.65,1.7,.55]]){const svg=document.getElementById('complex-svg');const before=svg.innerHTML;for(const [key,value] of [['alpha',a],['omega',w],['dt',d]]){const input=document.getElementById('sim-'+key);input.value=value;input.dispatchEvent(new Event('input',{bubbles:true}));}results.push({a,w,d,re:+svg.dataset.finalReal,im:+svg.dataset.finalImag,changed:before!==svg.innerHTML});}return results})()`);
for(const v of cases){assert(Math.abs(v.re-Math.exp(v.a*v.d*30)*Math.cos(v.w*v.d*30))<1e-10);assert(Math.abs(v.im-Math.exp(v.a*v.d*30)*Math.sin(v.w*v.d*30))<1e-10);}
assert(cases.slice(1).every(x=>x.changed));
await evaluate("document.getElementById('sim-alpha').focus()");const before=await evaluate("+document.activeElement.value");await c('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});await c('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});assert(await evaluate('+document.activeElement.value')>before);
await evaluate("document.getElementById('complex-svg').scrollIntoView({block:'center',behavior:'instant'})");writeFileSync(join(out,'guide-simulator.png'),Buffer.from((await c('Page.captureScreenshot',{format:'png'})).data,'base64'));
for(const section of ['architecture','ssd','data','training','optimization','evidence']){
await c('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
await evaluate(`document.getElementById('${section}').scrollIntoView({behavior:'instant'})`);
await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
writeFileSync(join(out,`guide-${section}.png`),Buffer.from((await c('Page.captureScreenshot',{format:'png'})).data,'base64'));
}
assert.equal(errors.length,0);
const bytes=readFileSync(file);const receipt={status:'pass',artifact:{sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length},offline:true,viewports:checks,links:anchors.length,recurrenceCases:cases,keyboard:'pass',runtimeErrors:errors};writeFileSync(join(out,'guide-browser.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
}finally{clearTimeout(timeout);chrome.kill();}
