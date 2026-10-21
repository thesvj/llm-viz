import React from 'react';
import clsx from 'clsx';
import { useProgramState } from '../Sidebar';
import { explainCell } from '../CellExplain';
import { DimStyle, dimStyleText } from '../walkthrough/WalkthroughTools';
import { IBlkDef } from '../GptModelLayout';
import { Vec3 } from '@/src/utils/vector';
import { animCursor } from '../flow/flowAnim';
import { flowActive } from '../flow/flowState';

const dimMeaning: Partial<Record<DimStyle, string>> = {
    [DimStyle.T]: 'token position',
    [DimStyle.t]: 'token position',
    [DimStyle.C]: 'embedding channel',
    [DimStyle.A]: 'head channel',
    [DimStyle.n_vocab]: 'vocabulary entry',
    [DimStyle.C4]: 'MLP hidden unit',
};

function fmt(v: number | null | undefined) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    let a = Math.abs(v);
    if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(3);
    return v.toFixed(4);
}

function fmtShort(v: number) {
    let a = Math.abs(v);
    if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(1);
    return v.toFixed(3);
}

/** The cell being inspected: the hovered one, else the pinned one. */
export function getInspectTarget(state: ReturnType<typeof useProgramState>): { blk: IBlkDef, idx: Vec3, isPinned: boolean } | null {
    let hover = state.display.hoverTarget;
    let pinned = state.display.pinned;
    if (hover) {
        return { blk: hover.mainCube, idx: hover.mainIdx, isPinned: false };
    }
    if (pinned && state.layout.cubes[pinned.cubeIdx]) {
        return { blk: state.layout.cubes[pinned.cubeIdx], idx: pinned.idx, isPinned: true };
    }
    // during guide playback, follow the number being computed
    let cur = flowActive(state) ? animCursor(state) : null;
    if (cur) {
        return { blk: cur.blk, idx: cur.idx, isPinned: false };
    }
    return null;
}

function kindOf(blk: IBlkDef) {
    return blk.t === 'w' ? { label: 'learned weight', cls: 'bg-indigo-100 text-indigo-800' }
        : blk.t === 'i' ? { label: 'activation', cls: 'bg-green-100 text-green-800' }
        : { label: 'helper (aggregate)', cls: 'bg-amber-100 text-amber-800' };
}

export const Inspector: React.FC = () => {
    let state = useProgramState();
    let target = getInspectTarget(state);

    if (!target) {
        return <div className='bg-white/95 rounded-lg shadow border border-gray-200 text-sm px-3 py-2 text-gray-500'>
            Hover any block to inspect a number. Click to pin it.
        </div>;
    }
    let { blk, idx, isPinned } = target;

    let ex = explainCell(blk, idx, state.layout.tokEmbedObj, state.layout.idxObj);
    let kind = kindOf(blk);
    let dims = [
        blk.dimX !== DimStyle.None ? { d: blk.dimX, n: blk.cx, i: idx.x } : null,
        blk.dimY !== DimStyle.None ? { d: blk.dimY, n: blk.cy, i: idx.y } : null,
    ].filter(Boolean) as { d: DimStyle, n: number, i: number }[];

    let match = ex && ex.stored !== null && ex.recomputed !== null
        ? Math.abs(ex.stored - ex.recomputed) <= 1e-3 * Math.max(1, Math.abs(ex.stored)) : null;

    return <div className='bg-white/95 rounded-lg shadow border border-gray-200 text-sm max-h-[60vh] overflow-y-auto overflow-x-hidden pointer-events-auto'>
        <div className='px-3 py-2 border-b border-gray-100 flex items-center gap-2'>
            <span className='font-semibold text-gray-900 truncate'>{blk.name || 'block'}</span>
            <span className={clsx('text-[10px] px-1.5 py-0.5 rounded', kind.cls)}>{kind.label}</span>
            {isPinned && <span className='ml-auto text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-white'>pinned</span>}
        </div>
        <div className='px-3 py-2 flex flex-col gap-1'>
            <div className='text-xs text-gray-600'>
                Shape {dims.map(d => `${dimStyleText(d.d)}=${d.n}`).join(' × ') || 'scalar'}
            </div>
            <div className='text-xs text-gray-600'>
                Cell {dims.map((d, k) => <span key={k} className='mr-2'><span className='font-mono'>{dimStyleText(d.d)}={d.i}</span>{dimMeaning[d.d] ? ` (${dimMeaning[d.d]})` : ''}</span>)}
            </div>
            <div className='flex items-baseline gap-2 mt-1'>
                <span className='text-xs text-gray-500'>value</span>
                <span className='font-mono text-lg text-gray-900'>{fmt(ex?.stored)}</span>
            </div>
        </div>
        {ex && <div className='px-3 py-2 border-t border-gray-100 flex flex-col gap-1'>
            <div className='font-semibold text-gray-800'>{ex.op}</div>
            {ex.formula && <div className='font-mono text-xs bg-gray-50 rounded px-2 py-1 text-gray-800 break-words'>{ex.formula}</div>}
            {ex.words && <div className='text-xs text-gray-600'>{ex.words}</div>}
            {ex.legend && <div className='text-[11px] text-gray-500'>{ex.legend}</div>}
            {ex.terms.length > 0 && <table className='text-xs font-mono mt-1 w-full table-fixed'>
                <thead><tr className='text-gray-500'>{ex.termHeader.map(h => <th key={h} className='text-right first:text-left font-normal pr-1 truncate'>{h}</th>)}</tr></thead>
                <tbody>
                    {ex.terms.map((t, i) => <tr key={i}>
                        <td className='pr-1 text-gray-500 truncate'>{t.label}</td>
                        {t.values.map((v, j) => <td key={j} className='pr-1 text-right whitespace-nowrap'>{v === null ? '' : fmtShort(v)}</td>)}
                    </tr>)}
                    {ex.termCount > ex.terms.filter(t => t.label !== 'bias').length && <tr><td colSpan={4} className='text-gray-400'>… {ex.termCount} terms in total</td></tr>}
                </tbody>
            </table>}
            {ex.recomputed !== null && <div className={clsx('text-xs mt-1', match ? 'text-green-700' : 'text-red-700')}>
                {match ? '✓' : '✗'} recomputed from the inputs above: <span className='font-mono'>{fmt(ex.recomputed)}</span>
                {!match && ' (does not match the stored value)'}
            </div>}
        </div>}
        <div className='px-3 py-1.5 border-t border-gray-100 text-[11px] text-gray-400'>
            {isPinned ? 'Click empty space to unpin.' : 'Click to pin this cell.'}
        </div>
    </div>;
};
