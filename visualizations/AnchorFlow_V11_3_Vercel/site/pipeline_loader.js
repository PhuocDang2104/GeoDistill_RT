/* Lossless gzip+byteplane+XOR transport. Native maps remain exact Float32. */
window.PipelineLoader={decode:async (pack,binary)=>{
  if(pack.codec!=='float32-byteplane-gzip-v1')throw new Error('Unknown web trace codec');
  if(typeof DecompressionStream!=='function')throw new Error('Use a recent Chrome/Edge with gzip DecompressionStream support');
  let compressed=binary&&new Uint8Array(binary);
  if(!compressed){const raw=atob(pack.payload);compressed=new Uint8Array(raw.length);for(let k=0;k<raw.length;k++)compressed[k]=raw.charCodeAt(k);}
  if(pack.payload_sha256){const hash=new Uint8Array(await crypto.subtle.digest('SHA-256',compressed));const value=Array.from(hash,b=>b.toString(16).padStart(2,'0')).join('');if(value!==pack.payload_sha256)throw new Error('Full jet data checksum mismatch');}
  const buffer=await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const bytes=new Uint8Array(buffer),arrays={};
  for(const [name,d]of Object.entries(pack.arrays)){
    const encoded=bytes.subarray(d.offset,d.offset+d.bytes);let result;
    if(d.type==='u1')result=new Uint8Array(encoded);
    else{
      const n=d.bytes/4,out=new Uint8Array(d.bytes);for(let b=0;b<4;b++)for(let k=0;k<n;k++)out[k*4+b]=encoded[b*n+k];
      const words=new Uint32Array(out.buffer);
      if(d.temporal_stride)for(let k=d.temporal_stride;k<n;k++)words[k]^=words[k-d.temporal_stride];
      if(d.reference){const ref=arrays[d.reference],u=new Uint32Array(ref.data.buffer),h=d.shape.at(-2),w=d.shape.at(-1),rh=ref.shape.at(-2),rw=ref.shape.at(-1);
        for(let y=0;y<h;y++)for(let x=0;x<w;x++)words[y*w+x]^=u[Math.floor(y*rh/h)*rw+Math.floor(x*rw/w)];}
      result=new Float32Array(out.buffer);
    }
    if(result.length!==d.shape.reduce((a,b)=>a*b,1))throw new Error('Trace shape mismatch: '+name);
    arrays[name]={data:result,shape:d.shape};
  }
  if(bytes.length!==pack.meta.web_contract.lossless_bytes)throw new Error('Incomplete decompression');
  delete pack.payload;return {meta:pack.meta,arrays,compact:true};
}};
