import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('D:/hwd/conv/output/Conv_method_editable.pptx'));
const b=await p.export({slide:p.slides.items[0],format:'png',scale:1});
await fs.writeFile('D:/hwd/conv/.pptx-build/final-preview.png',new Uint8Array(await b.arrayBuffer()));
console.log((await p.inspect({kind:'slide',maxChars:1000})).ndjson);
