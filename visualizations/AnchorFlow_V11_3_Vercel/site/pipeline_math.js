/* Offline native MathML; operator names use roman glyphs, no TeX macro parser. */
(() => {
  const i=x=>`<mi>${x}</mi>`,n=x=>`<mn>${x}</mn>`,o=x=>`<mo>${x}</mo>`,r=(...x)=>`<mrow>${x.join('')}</mrow>`;
  const sub=(x,y)=>`<msub>${x}${y}</msub>`,sup=(x,y)=>`<msup>${x}${y}</msup>`,frac=(x,y)=>`<mfrac>${x}${y}</mfrac>`;
  const fn=(x,y)=>r(`<mi mathvariant="normal">${x}</mi>`,o('('),y,o(')'));
  const math=x=>`<div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block">${x}</math></div>`;
  const z=i('z'),v=i('v'),t=i('τ'),s=i('s'),u=i('t'),j=i('j'),f=sub(i('f'),sub(i('θ'),i('c'))),fine=sub(i('f'),sub(i('θ'),i('f')));
  const cards=[
    ['1. Jet = một local surface',
     math(r(j,o('='),o('['),v,o(','),sub(i('g'),i('x')),o(','),sub(i('g'),i('y')),o(','),sub(i('h'),i('xx')),o(','),sub(i('h'),i('xy')),o(','),sub(i('h'),i('yy')),o(']')))+
     math(r(fn('v',r(s,o(','),u)),o('='),v,o('+'),sub(i('g'),i('x')),s,o('+'),sub(i('g'),i('y')),u,o('+'),frac(n(1),n(2)),sup(i('δ'),i('T')),i('H'),i('δ'))),
     'δ = [s,t]. Value v = 1/D, slope g và Hessian H mô tả mặt quadratic quanh cell. Trục s,t theo grid; height m⁻¹, không phải world coordinates.'],
    ['2. Coarse NODE tại D4',
     math(r(frac(r(i('d'),z),r(i('d'),t)),o('='),f,o('('),t,o(','),z,o(')'),o('='),n(2),fn('tanh',frac(i('A'),r(n(2),v,i('κ'))))))+
     math(r(fn('z',n(1)),o('='),fn('z',n(0)),o('+'),`<msubsup>${o('∫')}${n(0)}${n(1)}</msubsup>`,f,i('d'),t)),
     'A = learned reaction + sparse source + analytic transported neighbors. Refresh current state mỗi RHS call. Bosh3 điều khiển numerical steps; T=1, half-time dense output, không learned h/stop. Phép chia theo channel.'],
    ['3. Fine NODE tại D2',
     math(r(sub(i('k'),n(1)),o('='),fine,o('('),n(0),o(','),sup(z,n(0)),o(')')))+
     math(r(sub(i('z'),i('mid')),o('='),sup(z,n(0)),o('+'),frac(n(1),n(2)),sub(i('k'),n(1))))+
     math(r(sup(z,n(1)),o('='),sup(z,n(0)),o('+'),fine,o('('),frac(n(1),n(2)),o(','),sub(z,i('mid')),o(')'))),
     'Seed lại trên half-grid; field fine riêng, shared weights giữa k₁/k₂. Midpoint h=1, đúng 2 NFE. Thanh progress dùng z(τ)=z₀+(τ−τ²)k₁+τ²k₂: continuous extension bậc 2, không thêm RHS. Dense half không đồng nhất với internal RK-mid và không exact ODE solution. Không adaptive tolerance/rejection.'],
    ['4. Readout → sensor fusion',
     math(r(sub(i('D'),i('query')),o('='),frac(n(1),r(sub(o('∑'),i('r')),sub(i('ω'),i('r')),sub(v,i('r'))))))+
     math(r(sub(i('D'),n(1)),o('='),sub(i('D'),i('prePIR')),o('+'),i('ΔD')))+
     math(r(sub(i('D'),i('full')),o('='),o('('),n(1),o('−'),i('g'),o(')'),sub(i('D'),n(1)),o('+'),i('g'),i('S'))),
     'Five-jet consensus trộn inverse depth. PIR dùng phase-preserving donor sparse residual, không GT. Sensor gate g đã bao gồm mask và learned reliability; không hard anchor bắt buộc.']
  ];
  document.getElementById('theoryCards').innerHTML=cards.map(([title,equations,body])=>`<article><h3>${title}</h3>${equations}<p>${body}</p></article>`).join('');
})();
