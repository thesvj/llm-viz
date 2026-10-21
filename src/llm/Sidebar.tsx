import s from './Sidebar.module.scss';
import React, { createContext, useContext, useState } from 'react';
import clsx from 'clsx';
import { IPhaseDef } from './walkthrough/WalkthroughTools';
import { PhaseTimeline } from './PhaseTimeline';
import { Commentary } from './Commentary';
import { IProgramState } from './Program';
import { Popup, PopupPos } from '@/src/utils/Portal';
import { useSubscriptions } from '../utils/hooks';
import { FlowGuide } from './flow/FlowGuide';
import { flowActive, goToStep } from './flow/flowState';
import { PresetFacts } from './flow/PresetFacts';

export const WalkthroughSidebar: React.FC = () => {
    let progState = useProgramState();
    let walkthrough = progState.walkthrough;
    let camera = progState.camera;
    let [menuVisible, setMenuVisible] = useState(false);
    let [menuButtonEl, setMenuButtonEl] = useState<HTMLElement | null>(null);

    function handlePhaseClick(ev: React.MouseEvent, phase: IPhaseDef) {
        if (walkthrough.phase !== phase.id) {
            walkthrough.phase = phase.id;
            walkthrough.time = 0;
            walkthrough.running = false;
            progState.markDirty();
        }
        setMenuVisible(false);
        ev.preventDefault();
    }

    function stepModel() {
        console.log('stepping model');
        progState.stepModel = true;
        progState.markDirty();
    }

    let menu = <>
        <div className={s.topSplit}>
            <div className={s.toc}>
                {walkthrough.phaseList.map((group, i) => {

                    return <div key={group.groupId} className={s.phaseGroup}>
                        <div className={s.phaseGroupTitle}>{group.title}</div>

                        {group.phases.map((phase, j) => {
                            let active = walkthrough.phase === phase.id;

                            return <div key={phase.id} className={clsx(s.phase, active && s.active)} onClick={ev => handlePhaseClick(ev, phase)}>
                                <div className={s.phaseTitle}>{phase.title}</div>
                            </div>;
                        })}
                    </div>;
                })}
            </div>
            {/* <div className={s.helpers}>
                <div className={s.camStats}>
                    (center, center) =
                </div>
                <div className={s.camStats}>
                    new {camera.center.toString(1)}, new {camera.angle.toString(1)}
                </div>
            </div> */}
        </div>
    </>;

    let preset = progState.examples[progState.currExampleId];
    if (preset?.preset) {
        let back = () => {
            let mode = progState.modelMode === 'gpt2' ? 'llama' : progState.modelMode;
            let rt = progState.modern[mode];
            progState.currExampleId = -1;
            if (rt) {
                progState.modelMode = mode;
                progState.flow.token = rt.inputLen - 1;
                // the modern layout is rebuilt on the next frame; frame the current step after that
                setTimeout(() => goToStep(progState, progState.flow.step), 100);
            }
            progState.markDirty();
        };
        return <div className={s.walkthrough}>
            <div className={s.content}><PresetFacts name={preset.name} sub={preset.sub} config={preset.preset} onBack={back} /></div>
        </div>;
    }

    if (flowActive(progState)) {
        return <div className={s.walkthrough}>
            <div className={s.content}><FlowGuide /></div>
        </div>;
    }

    return <div className={s.walkthrough}>
        <div className={s.split}>

            <div className={s.timelineLeft}>
                <PhaseTimeline />
            </div>

            <div className={s.content}>
                {/* <div className={s.menuTopBar}>
                    <div className={s.menu} ref={setMenuButtonEl} onClick={() => setMenuVisible(a => !a)}>Menu &gt;</div>
                    {menuVisible && <Popup targetEl={menuButtonEl} placement={PopupPos.BottomLeft} className={s.mainMenu} closeBackdrop onClose={() => setMenuVisible(false)}>
                        {menu}
                    </Popup>}
                    <div onClick={() => stepModel()}>Step</div>
                </div> */}
                <Commentary />
            </div>

        </div>
    </div>;
};

export let ProgramStateContext = createContext<IProgramState>(null!);

export function useProgramState() {
    let context = useContext(ProgramStateContext);
    useSubscriptions(context?.htmlSubs);
    return context;
}
