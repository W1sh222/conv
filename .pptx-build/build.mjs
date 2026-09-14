import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const root='D:/hwd/conv';
process.env.RUNTIME_NODE_MODULES='C:/Users/W1sh222/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const skill='C:/Users/W1sh222/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs'));
const font=resolvePresentationFont({fontFamily:'Arial'});
const p=Presentation.create({slideSize:{width:1860,height:845}});
const s=p.slides.add();s.background.fill='#FFFFFF';
const colors={blue:'#D2EFF9',orange:'#F7C49F',gray:'#ECECEC',ink:'#121212'};
let count=0;
function rect(name,x,y,w,h,fill,round=0,stroke=2){return s.shapes.add({name,geometry:round?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:colors.ink,width:stroke},...(round?{borderRadius:round}:{})});}
function txt(name,text,x,y,w,h,size=26,rot=0){const a=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h,rotation:rot},fill:'none',line:{fill:'none',width:0}});a.text=text;a.text.style={typeface:font,fontSize:size,bold:true,color:colors.ink,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};return a;}
function line(name,x1,y1,x2,y2,width=3,dash=false){const x=Math.min(x1,x2),y=Math.min(y1,y2),w=Math.max(Math.abs(x2-x1),.1),h=Math.max(Math.abs(y2-y1),.1);return s.shapes.add({name,geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:colors.ink,width,style:dash?'dashed':'solid'},customPaths:[{width:w,height:h,commands:[{moveTo:{x:x1-x,y:y1-y}},{lineTo:{x:x2-x,y:y2-y}}]}]});}
function arrow(name,pts,width=4,dash=false,head=13){for(let i=1;i<pts.length;i++)line(name+' segment '+i,...pts[i-1],...pts[i],width,dash);let [x,y]=pts.at(-1),[a,b]=pts.at(-2);let dx=x-a,dy=y-b,l=Math.hypot(dx,dy);dx/=l;dy/=l;const points=[[x,y],[x-head*dx-head*.48*dy,y-head*dy+head*.48*dx],[x-head*dx+head*.48*dy,y-head*dy-head*.48*dx]];const xx=Math.min(...points.map(z=>z[0])),yy=Math.min(...points.map(z=>z[1])),ww=Math.max(...points.map(z=>z[0]))-xx,hh=Math.max(...points.map(z=>z[1]))-yy;s.shapes.add({name:name+' arrowhead',geometry:'custom',position:{left:xx,top:yy,width:ww,height:hh},fill:colors.ink,line:{fill:'none',width:0},customPaths:[{width:ww,height:hh,commands:[{moveTo:{x:points[0][0]-xx,y:points[0][1]-yy}},...points.slice(1).map(z=>({lineTo:{x:z[0]-xx,y:z[1]-yy}})),{close:{}}]}]});}
const pal=['#FAFCFC','#D2EAF7','#A8D8F2','#7CC3EB','#51AFE4','#2589D1','#2375B9'];
function grid(name,x,y,cols,rows,cw,ch,fn){for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)rect(`${name} row ${r+1} col ${c+1}`,x+c*cw,y+r*ch,cw,ch,fn(r,c),0,.8);}
function heat(name,x,y,w,h){grid(name,x,y,6,6,w/6,h/6,(r,c)=>pal[Math.max(0,6-Math.abs(r-c)-((r+c)%3===0?0:1))]);line(name+' diagonal',x,y,x+w,y+h,.6);}
function box(name,text,x,y,w,h,size=30,fill=colors.gray){rect(name,x,y,w,h,fill,14,2.5);txt(name+' label',text,x+4,y+2,w-8,h-4,size);}

// Group surfaces.
rect('Score Estimation background',14,315,968,335,colors.blue,30,2.4);
rect('Refinement background',994,315,515,335,colors.blue,30,2.4);
txt('Score Estimation vertical label','Score Estimation',-103,463,300,45,35,270);
txt('Refinement vertical label','Refinement',910,463,230,44,35,270);
const cards=[[83,370,133,209,'Sample'],[231,370,165,209,'Dot Product'],[411,370,178,209,'Normalize'],[604,370,180,209,'Aggregate']];
for(const [x,y,w,h,title] of cards){rect(title+' card',x,y,w,h,colors.orange,14,2);txt(title+' caption',title,x+4,535,w-8,38,24);}

// Native editable mini-diagrams.
grid('Sampling',99,405,5,5,20.2,20.6,(r,c)=>c===4-r?pal[6]:pal[(r+c)%2]);
for(const [a,b] of [[99,119],[122,155],[160,200]]){line('Stride bracket left '+a,a,512,a,521,2);line('Stride bracket base '+a,a,521,b,521,2);line('Stride bracket right '+a,b,521,b,512,2);}
line('Inverse antidiagonal',99,508,200,405,.7);
txt('Q mini label','Q',247,380,34,27,26);
txt('K transpose mini label','Kᵀ',330,381,45,27,26);
grid('Dot Q',246,411,2,4,16,19,(r,c)=>pal[2+(r+c)%3]);
txt('Multiplication symbol','×',282,428,27,33,31);
grid('Dot K transpose',312,418,4,2,17,20,(r,c)=>pal[2+(r+c)%3]);
arrow('Dot output arrow',[[313,466],[313,486]],2,false,8);
grid('Dot output',287,489,3,3,15.5,16,(r,c)=>pal[2+(r+c)%4]);

grid('Causal mask',424,409,4,4,18,21,(r,c)=>c>r?'#F5F5F5':pal[6-Math.abs(r-c)]);
for(let r=0;r<4;r++)for(let c=r+1;c<4;c++){const x=424+c*18+3,y=409+r*21+3;line('Masked cross '+r+c+' a',x,y,x+12,y+15,1.5);line('Masked cross '+r+c+' b',x+12,y,x,y+15,1.5);}
arrow('Normalize arrow',[[498,455],[513,455]],2,false,7);
grid('Normalized probabilities',516,443,4,1,16,22,(r,c)=>pal[5-c]);
txt('Probability sum','Σ = 1',518,470,63,30,24);

grid('Fine block grid',616,418,4,4,17.5,18.5,(r,c)=>pal[2+(r+c)%4]);
for(let r=0;r<2;r++)for(let c=0;c<2;c++){
 const x=612+c*39,y=412+r*43;
 for(const [a,b,d,e] of [[x,y,x+36,y],[x+36,y,x+36,y+39],[x+36,y+39,x,y+39],[x,y+39,x,y]])line(`Block outline ${r}${c} ${a}-${b}`,a,b,d,e,2,true);
}
txt('Block sum sigma','Σ',699,429,27,34,29);
arrow('Aggregation arrow',[[701,472],[721,472]],2,false,8);
grid('Coarse block grid',730,432,2,2,23,26,(r,c)=>pal[3+(r+c)%3]);

heat('Initial score heatmap',809,379,144,147);
txt('Initial Score Map caption','Initial Score Map',792,539,178,40,23);
box('Convolution','7 × 7 Conv',1062,416,142,94,25,colors.orange);
heat('Refined score heatmap',1222,379,136,147);
txt('Refined Scores caption','Refined Scores',1200,539,180,40,24);
box('Top-K','Top-K',1385,416,103,91,29);
for(const [a,b] of [[216,231],[396,411],[589,604],[784,807],[953,991],[1204,1220],[1358,1383]])arrow('Processing flow '+a,[[a,462],[b,462]],4,false,12);

// Original inputs and shared RoPE branch.
box('Q input','Q',15,118,88,69,35);
box('K input','K',15,212,88,68,35);
box('RoPE','RoPE',178,157,149,82,33,colors.orange);
line('Q to junction',103,153,143,153,4);line('QK junction',143,153,143,246,4);line('K to junction',103,246,143,246,4);
arrow('QK into RoPE',[[143,199],[176,199]],4);
arrow('RoPE to score estimator',[[251,239],[251,313]],4);
arrow('Shared RoPE QK',[[327,199],[1657,199],[1657,425]],3,true,14);
txt('Shared branch label','Full-resolution RoPE Q/K',754,147,434,46,29);
box('V input','V',1762,117,86,68,35);
arrow('V to attention',[[1805,185],[1805,425]],4);
box('Block Sparse Flash-Attn','Block Sparse\nFlash-Attn',1577,427,271,153,33);
arrow('Sparse mask to kernel',[[1488,462],[1527,462],[1527,504],[1575,504]],4,false,14);
arrow('Sparse index annotation',[[1545,504],[1545,602]],2.4,false,11);
txt('Sparse Block Index label','Sparse Block Index',1516,602,232,40,23);
arrow('Attention output flow',[[1759,580],[1759,676]],4,false,14);
box('Attention Output','Attention Output',1604,678,244,77,26);
s.speakerNotes.textFrame.setText('Reconstructed from the user-provided Conv method diagram. All visible diagram elements are native editable PowerPoint shapes, text boxes, paths, and matrix cells.');
await fs.writeFile(root+'/.pptx-build/preview.png',new Uint8Array(await (await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
await (await PresentationFile.exportPptx(p)).save(root+'/.pptx-build/candidate.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:root+'/.pptx-build/candidate.pptx',finalPath:root+'/output/Conv_method_editable.pptx',pythonExecutable:'C:/Users/W1sh222/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu',`${1860*9525},${845*9525}`],fontPolicy:{basis:'design',families:[font]},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/.pptx-build/validation.json'});
console.log(JSON.stringify(result));
