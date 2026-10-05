const assert=require('node:assert/strict');
// Browser script is intentionally not an ES module. Run it in a browser-like
// context so tests also work inside this project's type=module package.
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const browser={};
vm.runInNewContext(fs.readFileSync(path.join(__dirname,'jet_math.js'),'utf8'),browser);
const M=browser.JetMath;
const close=(a,b,t=1e-10)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
let checks=0;
for(let n=0;n<100;n++){
  const j=[.1,.003,-.004,.002,-.001,.003],dx=n/100-.5,dy=.3,s=.13,t=-.22;
  const moved=M.translate(j,dx,dy);
  close(M.polynomial(moved,s,t),M.polynomial(j,s+dx,t+dy));
  const moved2=M.translate(moved,-dx,-dy);moved2.forEach((x,i)=>close(x,j[i]));checks+=7;
}
const j=[.08,.004,-.002,.001,-.002,.003];M.decode(M.encode(j)).forEach((x,i)=>close(x,j[i]));checks+=6;
for(const zz of [-100,-10,0,10,100]){const d=M.decode(Array(6).fill(zz));assert.ok(d[0]>=1/120-1e-10&&d[0]<=10+1e-10);for(let i=1;i<6;i++)assert.ok(Math.abs(d[i])<=M.kappa[i]*d[0]+1e-10);checks+=6;}
const z=M.encode(j),f=[.2,-.1,.04,.08,-.03,.06],vel=M.jetVelocity(z,f),eps=1e-6;
const plus=M.decode(z.map((x,i)=>x+eps*f[i])),minus=M.decode(z.map((x,i)=>x-eps*f[i]));
vel.forEach((x,i)=>close(x,(plus[i]-minus[i])/(2*eps),1e-8));checks+=6;
let calls=0;const rk=M.midpoint((t,z)=>{calls++;return z;},[1]);assert.equal(calls,2);close(rk.mid[0],1.5);close(rk.end[0],2.5);checks+=3;
const jets=[j,...[[1,0],[-1,0],[0,1],[0,-1]].map(([x,y])=>M.translate(j,x,y))];
for(const s of [-.25,.25])for(const t of [-.25,.25]){const c=M.consensus(jets,[.5,.125,.125,.125,.125],s,t);c.values.forEach(x=>close(x,M.polynomial(j,s,t)));close(c.depth,1/M.polynomial(j,s,t));checks+=6;}
const harmonic=M.consensus([[.1,0,0,0,0,0],[.05,0,0,0,0,0],j,j,j],[.5,.5,0,0,0],0,0);close(harmonic.depth,1/.075);assert.notEqual(harmonic.depth,15);checks+=2;
for(const t of [0,.13,.5,.87,1]){
  close(M.midpointDense([3],[4],[5],t)[0],3+2*t); // constant field
  close(M.midpointDense([0],[0],[1],t)[0],t*t); // z'=2t
  checks+=2;
}
close(M.midpointDense([1],rk.mid,rk.end,.5)[0],1.625);
assert.notEqual(M.midpointDense([1],rk.mid,rk.end,.5)[0],rk.mid[0]);checks+=2;
const errors=[.2,.1,.05].map(h=>Math.abs(M.midpointDense([1],[1+h/2],[1+h+h*h/2],.37)[0]-Math.exp(.37*h)));
assert.ok(errors[0]/errors[1]>7);assert.ok(errors[1]/errors[2]>7);checks+=2;
console.log(JSON.stringify({passed:true,numerical_checks:checks,contracts:['exact origin translation','bounded chart','decode Jacobian','midpoint exactly 2 RHS','phase origin signs','inverse-depth not metric-depth averaging']}));
