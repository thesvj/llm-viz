import React, { useState } from 'react';
import clsx from 'clsx';
import { useProgramState } from '../Sidebar';
import { getBlockValueAtIdx } from './DataFlow';
import { IBlkDef } from '../GptModelLayout';
import { Vec3 } from '@/src/utils/vector';

// Sequential "Blues" (ColorBrewer): 0 = white, 1 = dark blue. Probabilities use a fixed 0–1 scale.
const blues = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'];
function blueOf(p: number) {
    let t = Math.max(0, Math.min(1, p)) * (blues.length - 1);
    let i = Math.min(blues.length - 2, Math.floor(t));
    let f = t - i;
    let a = parseInt(blues[i].slice(1), 16), b = parseInt(blues[i + 1].slice(1), 16);
    let ch = (s: number) => Math.round(((a >> s) & 255) * (1 - f) + ((b >> s) & 255) * f);
    return `rgb(${ch(16)},${ch(8)},${ch(0)})`;
}

/** probs[q][k] for the first n tokens. */
function readHead(blk: IBlkDef, n: number): number[][] {
    let rows: number[][] = [];
    for (let q = 0; q < n; q++) {
        let row: number[] = [];
        for (let k = 0; k < n; k++) {
            row.push(k > q ? NaN : getBlockValueAtIdx(blk, new Vec3(k, q, 0)) ?? NaN);
        }
        rows.push(row);
    }
    return rows;
}

/** A short plain-words guess at what a head does, from where its attention mass goes. */
function headPattern(p: number[][]): string {
    let n = p.length;
    if (n < 2) return '';
    let self = 0, prev = 0, first = 0, cnt = 0;
    for (let q = 1; q < n; q++) {
        self += p[q][q] || 0;
        prev += p[q][q - 1] || 0;
        first += p[q][0] || 0;
        cnt++;
    }
    let opts: [string, number][] = [['looks at itself', self / cnt], ['looks at the previous token', prev / cnt], ['looks at the first token', first / cnt]];
    opts.sort((a, b) => b[1] - a[1]);
    return opts[0][1] > 0.45 ? `${opts[0][0]} (${Math.round(opts[0][1] * 100)}%)` : 'spreads attention';
}

interface IHoverCell { layer: number, head: number, q: number, k: number, p: number }

export const AttentionPanel: React.FC = () => {
    let state = useProgramState();
    let [open, setOpen] = useState(false);
    let [hover, setHover] = useState<IHoverCell | null>(null);

    let layout = state.layout;
    let heads = layout.attnHeads ?? [];
    let tokens = layout.model?.inputTokens.localBuffer;
    let n = Math.min(layout.model?.inputLen ?? 0, layout.shape.T);

    if (!open) {
        return <button className='absolute bottom-0 left-0 m-3 px-3 py-1.5 bg-white rounded-lg shadow border border-gray-200 text-sm hover:bg-blue-50'
            onClick={() => setOpen(true)}>Attention maps</button>;
    }

    if (!tokens || n === 0 || heads.length === 0) {
        return null;
    }

    let tok = (i: number) => layout.tokenText(tokens![i]);
    let cell = n <= 8 ? 16 : n <= 16 ? 12 : 8;
    let label = 12;

    function flyTo(blk: IBlkDef, q: number, k: number) {
        // pin the cell, and point the camera at the head's attention matrix
        state.display.pinned = { cubeIdx: blk.idx, idx: new Vec3(k, q, 0) };
        let center = state.camera.modelMtx.mulVec3Proj(new Vec3(blk.x + blk.dx / 2, blk.y + blk.dy / 2, blk.z + blk.dz / 2));
        state.camera.desiredCamera = { center, angle: new Vec3(270, 4.5, 2.5) };
        state.markDirty();
    }

    return <div className='absolute bottom-0 left-0 m-3 bg-white/95 rounded-lg shadow border border-gray-200 text-sm max-w-[calc(100%-22rem)] pointer-events-auto'>
        <div className='px-3 py-1.5 border-b border-gray-100 flex items-center gap-3'>
            <span className='font-semibold text-gray-800'>Attention</span>
            <span className='text-xs text-gray-500'>row = token that is reading · column = token it reads from · darker = more attention</span>
            <button className='ml-auto text-xs text-gray-500 hover:text-gray-900' onClick={() => setOpen(false)}>Hide</button>
        </div>
        <div className='px-3 py-2 overflow-x-auto'>
            <table className='border-separate border-spacing-x-3 border-spacing-y-1'>
                <thead>
                    <tr>
                        <th />
                        {heads[0].map((_, h) => <th key={h} className='text-xs font-normal text-gray-500 text-left'>head {h}</th>)}
                    </tr>
                </thead>
                <tbody>
                    {heads.map((layerHeads, l) => <tr key={l}>
                        <td className='text-xs text-gray-500 pr-1 align-middle whitespace-nowrap'>layer {l}</td>
                        {layerHeads.map((blk, h) => {
                            let p = readHead(blk, n);
                            let size = label + n * cell;
                            return <td key={h} className='align-top'>
                                <svg width={size} height={size} className='block cursor-pointer'
                                    onMouseLeave={() => setHover(null)}>
                                    {Array.from({ length: n }, (_, i) => <React.Fragment key={i}>
                                        <text x={label + i * cell + cell / 2} y={label - 3} fontSize={9} textAnchor='middle' fill='#6b7280'>{tok(i)}</text>
                                        <text x={label - 3} y={label + i * cell + cell / 2 + 3} fontSize={9} textAnchor='end' fill='#6b7280'>{tok(i)}</text>
                                    </React.Fragment>)}
                                    {p.map((row, q) => row.map((v, k) => {
                                        let masked = isNaN(v);
                                        let isHover = hover && hover.layer === l && hover.head === h && hover.q === q && hover.k === k;
                                        return <rect key={`${q}-${k}`} x={label + k * cell} y={label + q * cell} width={cell - 1} height={cell - 1}
                                            fill={masked ? '#f3f4f6' : blueOf(v)}
                                            stroke={isHover ? '#ef4444' : 'none'} strokeWidth={isHover ? 1.5 : 0}
                                            onMouseEnter={() => setHover(masked ? null : { layer: l, head: h, q, k, p: v })}
                                            onClick={() => !masked && flyTo(blk, q, k)} />;
                                    }))}
                                </svg>
                                <div className='text-[10px] text-gray-500 mt-0.5' style={{ maxWidth: size }}>{headPattern(p)}</div>
                            </td>;
                        })}
                    </tr>)}
                </tbody>
            </table>
        </div>
        <div className='px-3 py-1.5 border-t border-gray-100 text-xs text-gray-700 min-h-[1.75rem]'>
            {hover
                ? <>Layer {hover.layer}, head {hover.head}: token {hover.q} <b>{tok(hover.q)}</b> gives <b>{(hover.p * 100).toFixed(1)}%</b> of its attention to token {hover.k} <b>{tok(hover.k)}</b>. Click to fly there in 3D.</>
                : <span className='text-gray-400'>Hover a square for details. Grey squares are blocked: a token cannot look at later tokens.</span>}
        </div>
    </div>;
};
