import React from 'react';
import { useProgramState } from '../Sidebar';
import clsx from 'clsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExpand, faMagnifyingGlass } from '@fortawesome/free-solid-svg-icons';
import { Vec3 } from '@/src/utils/vector';
import { Mat4f } from '@/src/utils/matrix';
import { IModelMode, modelModeInfo } from '../arch/ModernModels';
import { genModernLayout } from '../arch/ModernLayout';
import { flyToBlocks } from '../flow/flowCamera';
import { saveFlowView } from '../flow/flowState';

export const ModelSelectorToolbar: React.FC<{
}> = () => {
    let progState = useProgramState();

    function onExpandClick() {
        if (progState.currExampleId === -1 && progState.layout.kind === 'modern') {
            flyToBlocks(progState, progState.layout.cubes);
            return;
        }
        let example = progState.examples[progState.currExampleId] ?? progState.mainExample;
        progState.camera.desiredCamera = example.camera;
        progState.markDirty();
    }

    function onMagnifyClick() {
        let example = progState.examples[progState.currExampleId] ?? progState.mainExample;
        let layout = example.layout ?? progState.layout;

        // new Vec3(3.347, 48.000, -2.634), new Vec3(270.000, 4.500, 1.199)

        // new Vec3(-1.771, 0.750, -4.470), new Vec3(270.000, 4.500, 0.739)

        let obj = layout.residual0;
        let modelTarget = new Vec3(obj.x, obj.y, obj.z);
        let modelMtx = progState.camera.modelMtx.mul(Mat4f.fromTranslation(example.offset))

        let center = modelMtx.mulVec3Proj(modelTarget);
        let zoom = progState.currExampleId === -1 ? 0.7 : 4;
        progState.camera.desiredCamera = {
            center, angle: new Vec3(270, 4.5, zoom),
        };
        progState.markDirty();

    }

    function setMode(mode: IModelMode) {
        let rt = mode !== 'gpt2' ? progState.modern[mode] : null;
        if (mode !== 'gpt2' && !rt) {
            return; // still loading
        }
        progState.modelMode = mode;
        progState.display.pinned = null;
        progState.currExampleId = -1;
        progState.flow.step = 0;
        progState.flow.layer = 0;
        if (rt) {
            progState.flow.token = rt.inputLen - 1;
            // frame the input and the first layer, at nano-gpt's viewing angle
            let lay = genModernLayout(rt.model.config, null);
            let b0 = lay.blocks[0];
            flyToBlocks(progState, [lay.idxObj, lay.tokEmbedObj, lay.residual0, b0.norm1.out, ...b0.heads.map(h => h.wq), b0.resid1]);
        } else {
            progState.camera.desiredCamera = progState.mainExample.camera;
        }
        saveFlowView(progState);
        progState.markDirty();
    }

    function showExample(egIndex: number) {
        let example = progState.examples[egIndex];
        example.enabled = true;
        progState.currExampleId = egIndex;
        progState.display.pinned = null;
        if (example.preset) {
            example.layout ??= genModernLayout(example.preset, null, new Vec3(), 8);
            flyToBlocks(progState, example.layout.cubes, undefined, 1.1, example.offset);
        } else {
            progState.camera.desiredCamera = example.camera;
        }
        progState.markDirty();
    }

    let liveModes: IModelMode[] = ['gpt2', 'llama', 'moe'];
    let activeLive = progState.currExampleId === -1;
    let info = modelModeInfo[progState.modelMode];
    let btn = 'px-2.5 py-1 text-sm cursor-pointer select-none transition-colors';
    let on = 'bg-blue-600 text-white';
    let off = 'bg-white text-gray-800 hover:bg-blue-50';
    let iconBtn = 'p-1.5 w-8 h-8 flex justify-center items-center bg-white rounded-md shadow-sm border border-gray-200 cursor-pointer hover:bg-blue-50 text-gray-700';

    return <div className='absolute top-0 left-0 m-2 flex flex-col gap-1.5 pointer-events-none'>
        <div className='flex flex-row items-end gap-3 pointer-events-auto'>
            <div>
                <div className='text-[11px] font-medium uppercase tracking-wide text-gray-500 mb-0.5'>Model (real numbers)</div>
                <div className='flex rounded-md shadow-sm border border-gray-200 overflow-hidden divide-x divide-gray-200'>
                    {liveModes.map(mode => {
                        let ready = mode === 'gpt2' || !!progState.modern[mode];
                        let active = activeLive && progState.modelMode === mode;
                        return <div key={mode} title={modelModeInfo[mode].sub} onClick={() => setMode(mode)}
                            className={clsx(btn, active ? on : off, !ready && 'opacity-50 cursor-wait')}>{modelModeInfo[mode].title}</div>;
                    })}
                </div>
            </div>
            <div>
                <div className='text-[11px] font-medium uppercase tracking-wide text-gray-500 mb-0.5'>Real sizes (shape only)</div>
                <select value={progState.currExampleId} onChange={ev => { let i = +ev.target.value; i < 0 ? setMode(progState.modelMode) : showExample(i); }}
                    className={clsx('h-[30px] rounded-md shadow-sm border border-gray-200 text-sm px-2 cursor-pointer', !activeLive ? 'bg-blue-600 text-white' : 'bg-white text-gray-800')}>
                    <option value={-1}>Choose a real model…</option>
                    <optgroup label='GPT-2 family'>
                        {progState.examples.map((eg, i) => !eg.preset && !eg.hidden && <option key={i} value={i}>{eg.name}</option>)}
                    </optgroup>
                    <optgroup label='Llama family'>
                        {progState.examples.map((eg, i) => eg.preset && !eg.hidden && <option key={i} value={i}>{eg.name}</option>)}
                    </optgroup>
                </select>
            </div>
            <div className='flex gap-1.5'>
                <div className={iconBtn} title='Show the whole model' onClick={onExpandClick}><FontAwesomeIcon icon={faExpand} /></div>
                <div className={iconBtn} title='Zoom in on the input' onClick={onMagnifyClick}><FontAwesomeIcon icon={faMagnifyingGlass} /></div>
            </div>
        </div>
        <div className='text-xs text-gray-600 pointer-events-auto'>
            {activeLive ? info.sub : (progState.examples[progState.currExampleId]?.sub ?? '')}
        </div>
    </div>;

};
