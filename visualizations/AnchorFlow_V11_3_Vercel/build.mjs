/* Zero dependency static build; full jet arrays retained losslessly. */
import {readFile,writeFile,mkdir,copyFile,readdir,stat,link} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.dirname(fileURLToPath(import.meta.url)),src=path.join(root,'site'),out=path.join(root,'dist');
const manifest=JSON.parse(await readFile(path.join(root,'source_manifest.json'),'utf8'));
for(const [name,info] of Object.entries(manifest.files)){
  // Preserve recorded asset integrity without locking normal frontend edits.
  if(!name.endsWith('.bin.gz')&&!name.endsWith('.png'))continue;
  const bytes=await readFile(path.join(root,name));
  if(bytes.length!==info.bytes||createHash('sha256').update(bytes).digest('hex')!==info.sha256)throw Error('Source changed: '+name);
}
async function copyTree(dir,target){await mkdir(target,{recursive:true});for(const item of await readdir(dir,{withFileTypes:true})){
  const from=path.join(dir,item.name),to=path.join(target,item.name);
  if(item.name==='web_traces'||item.name==='catalog.json')continue;
  if(item.isDirectory())await copyTree(from,to);else await copyFile(from,to);
}}
await copyTree(src,out);await copyFile(path.join(root,'README_DEPLOY.md'),path.join(out,'README_DEPLOY.md'));await mkdir(path.join(out,'web_traces'),{recursive:true});
const evidence=JSON.parse(await readFile(path.join(src,'catalog.json'),'utf8'));
if(evidence.scenes.length!==10)throw Error('Expected ten paired scenes');
for(const scene of evidence.scenes){
  scene.rgb_data_url='data:image/png;base64,'+(await readFile(path.join(src,scene.rgb))).toString('base64');
  const stem=path.basename(scene.trace,'.js'),dir=path.join(src,'web_traces');
  const pack=JSON.parse(await readFile(path.join(dir,stem+'.pack.json'),'utf8'));
  if(pack.meta.sample_id!==scene.meta.sample_id||pack.meta.checkpoint_sha256!==evidence.baseline.checkpoint.sha256)throw Error('Scene/checkpoint mismatch');
  const gzip=await readFile(path.join(dir,stem+'.bin.gz'));
  if(gzip.length!==pack.meta.web_contract.gzip_bytes)throw Error('Incomplete payload: '+stem);
  pack.payload=gzip.toString('base64');
  await writeFile(path.join(out,scene.trace),'window.JET_PIPELINE_PACK='+JSON.stringify(pack)+';\n');
  await copyFile(path.join(dir,stem+'.proof.json'),path.join(out,'web_traces',stem+'.proof.json'));
  // Binary full jets load on demand. Hard links save local disk; deployment sees regular files.
  const binaryName=stem+'.jets.bin.gz',target=path.join(out,'web_traces',binaryName);
  try{await stat(target);}catch{try{await link(path.join(dir,binaryName),target);}catch{await copyFile(path.join(dir,binaryName),target);}}
}
await writeFile(path.join(out,'pipeline_catalog.js'),'window.PIPELINE_SCENES='+JSON.stringify(evidence.scenes)+';\nwindow.PIPELINE_BASELINE='+JSON.stringify(evidence.baseline)+';\n');
async function size(dir){let total=0;for(const item of await readdir(dir,{withFileTypes:true})){const file=path.join(dir,item.name);total+=item.isDirectory()?await size(file):(await stat(file)).size;}return total;}
console.log(JSON.stringify({built:true,scenes:evidence.scenes.length,output:'dist',static_output_bytes:await size(out),recorded_assets_verified:true,external_dependencies:0},null,2));
