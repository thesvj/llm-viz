import React from 'react';
import clsx from 'clsx';
import { useProgramState } from '../Sidebar';
import { IVizSettings } from '../VizSettings';
import { Inspector, getInspectTarget } from './Inspector';

function fmt(v: number) {
    let a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 100 || a < 0.01) return v.toExponential(1);
    return v.toPrecision(3);
}

const divergingCss = 'linear-gradient(to right, #2166ac, #92c5de, #f7f7f7, #f4a582, #b2182b)';

export const ViewPanel: React.FC = () => {
    let state = useProgramState();
    let viz = state.viz;

    function set<K extends keyof IVizSettings>(key: K, value: IVizSettings[K]) {
        viz[key] = value;
        state.markDirty();
    }

    function Toggle<K extends keyof IVizSettings>({ k, value, label, title }: { k: K, value: IVizSettings[K], label: string, title: string }) {
        let active = viz[k] === value;
        return <button title={title}
            className={clsx('px-2 py-0.5 rounded text-xs border', active ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:bg-blue-50')}
            onClick={() => set(k, value)}>{label}</button>;
    }

    let blk = getInspectTarget(state)?.blk;
    let range = blk?.access?.range;

    return <div className='absolute top-0 right-0 m-3 w-[19rem] flex flex-col gap-2 pointer-events-none'>
    <div className='bg-white/95 rounded-lg shadow border border-gray-200 text-sm select-none pointer-events-auto'>
        <div className='px-3 py-2 border-b border-gray-100 font-semibold text-gray-800'>View</div>
        <div className='px-3 py-2 flex flex-col gap-1.5'>
            <div className='flex items-center justify-between'>
                <span className='text-gray-600'>Colour scale</span>
                <span className='flex gap-1'>
                    <Toggle k='colorScale' value='auto' label='Per block' title='Each block uses its full colour range; the legend shows the real values' />
                    <Toggle k='colorScale' value='original' label='Original' title="The original hand-picked scales" />
                </span>
            </div>
            <div className='flex items-center justify-between'>
                <span className='text-gray-600'>Colours</span>
                <span className='flex gap-1'>
                    <Toggle k='palette' value='diverging' label='Blue–red' title='Blue = negative, white = zero, red = positive' />
                    <Toggle k='palette' value='classic' label='Classic' title='Original: black = negative, block colour = positive' />
                </span>
            </div>
            <div className='flex items-center justify-between'>
                <span className='text-gray-600'>Focus on hover</span>
                <span className='flex gap-1'>
                    <Toggle k='focus' value={true} label='On' title='Dim everything except the hovered block, its inputs and its outputs' />
                    <Toggle k='focus' value={false} label='Off' title='Show everything' />
                </span>
            </div>
        </div>
        <div className='px-3 py-2 border-t border-gray-100'>
            {viz.palette === 'diverging' ? <>
                <div className='h-3 rounded' style={{ background: divergingCss }} />
                <div className='flex justify-between text-xs text-gray-600 mt-0.5 font-mono'>
                    <span>{range ? fmt(-range) : '−'}</span>
                    <span>0</span>
                    <span>{range ? fmt(range) : '+'}</span>
                </div>
            </> : <div className='text-xs text-gray-600'>black = negative · grey = zero · block colour = positive</div>}
            <div className='text-xs text-gray-500 mt-1'>
                {blk ? <><span className='font-semibold text-gray-800'>{blk.name || 'block'}</span>
                    {range ? <> · full colour = ±{fmt(range)}{viz.colorScale === 'original' ? ' (scale not matched)' : ''}</> : ' · no values'}</>
                    : 'Hover a block to see its value range.'}
            </div>
        </div>
    </div>
    <Inspector />
    </div>;
};
