/* Auditable scalar math. No learned model is hidden in the teaching demo. */
(function (root) {
  'use strict';
  const kappa = [1, .5, .5, .25, .25, .25];
  const lo = Math.log(1 / 120), span = Math.log(1200);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  function polynomial(j, s, t) {
    return j[0] + j[1]*s + j[2]*t + .5*j[3]*s*s + j[4]*s*t + .5*j[5]*t*t;
  }
  function translate(j, dx, dy) {
    return [polynomial(j, dx, dy), j[1]+j[3]*dx+j[4]*dy,
      j[2]+j[4]*dx+j[5]*dy, j[3], j[4], j[5]];
  }
  function decode(z) {
    const v = Math.exp(lo + span*sigmoid(z[0]));
    return [v, ...z.slice(1).map((x,i) => v*kappa[i+1]*Math.tanh(x))];
  }
  function encode(j) {
    const v = clamp(j[0], 1/120, 10);
    const q = clamp((Math.log(v)-lo)/span, 1e-6, 1-1e-6);
    return [Math.log(q/(1-q)), ...j.slice(1).map((x,i) => Math.atanh(clamp(x/(v*kappa[i+1]), -.9999, .9999)))];
  }
  function jetVelocity(z, f) {
    const j = decode(z), q = sigmoid(z[0]);
    const dv = j[0]*span*q*(1-q)*f[0];
    return [dv, ...z.slice(1).map((x,i) => kappa[i+1]*(dv*Math.tanh(x)+j[0]*(1-Math.tanh(x)**2)*f[i+1]))];
  }
  function midpoint(field, z, h=1) {
    const k1 = field(0,z), mid = z.map((x,i) => x+.5*h*k1[i]);
    const k2 = field(.5*h,mid), end = z.map((x,i) => x+h*k2[i]);
    return {k1,mid,k2,end,nfe:2};
  }
  function midpointDense(initial, internalMid, terminal, t) {
    // Second-order continuous extension of explicit midpoint, not an exact ODE solution.
    // k1=2*(internalMid-initial), k2=terminal-initial; no extra RHS evaluation.
    return initial.map((v,k)=>v+2*(t-t*t)*(internalMid[k]-v)+t*t*(terminal[k]-v));
  }
  function consensus(jets, weights, s, t) {
    const offsets = [[0,0],[1,0],[-1,0],[0,1],[0,-1]];
    const values = jets.map((j,i) => clamp(polynomial(j,s-offsets[i][0],t-offsets[i][1]),1/120,10));
    const total = weights.reduce((a,b)=>a+b,0);
    if (!(total>0)) throw new Error('Consensus needs positive weight mass');
    const normalized = weights.map(w=>w/total);
    const v = values.reduce((sum,x,i)=>sum+x*normalized[i],0);
    return {values,weights:normalized,v,depth:1/v};
  }
  function teachingField(t,z) {
    // State/time-dependent analytic RHS, NOT the trained V11/V11_2 CNN.
    const j = decode(z), target = 1/12;
    const forcing = [1.5*(target-j[0]), -.3*j[1], -.3*j[2], -.5*j[3], -.5*j[4], -.5*j[5]];
    return forcing.map((a,i)=>2*Math.tanh(a/(2*j[0]*kappa[i]))*(1+.1*t));
  }
  const api = {kappa,clamp,sigmoid,polynomial,translate,encode,decode,jetVelocity,midpoint,midpointDense,consensus,teachingField};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.JetMath = api;
})(typeof window !== 'undefined' ? window : globalThis);
