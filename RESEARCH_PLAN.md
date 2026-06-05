thesis plan - ontology guided marl
(3 weeks left, ~1 run a day, no parallel. cutting hard)

main Q: does ontology knowledge actually help marl for cooperative household tasks

whats in / whats out
- SQ1 effectiveness - ont vs vanilla baseline -> KEEP, this is the whole point
- SQ3 coordination - shared abox (prox=5) vs independent (prox=0) -> KEEP, basically free
- SQ5 generalisation - move to collect_trash -> KEEP, 1 run
- SQ2 which component matters (prior/connectivity/semantic) -> DROP. future work. needs ablation code + like 3 more runs i dont have time for

what i actually have (old plan was wrong about most of this)
- its MAPPO not plain ppo. centralized critic, decentralized actors (ctde). mappo.py / mappo_policy.py, run from train_ppo.py
- curriculum, 4 stages: banana -> +mango -> +orange+grapes -> all 8. steps 2000/2000/3000/4000. advances at 80% over 100 eps
- env 32x23, 10 rooms, collect_food = 8 items (not 16, thats collect_trash), 3 agents, drop in kitchen
- obs size depends on n items (env_sb3 line 79) so 8 vs 16 dont match -> cant just load food weights into trash, important for SQ5
- hyperparams: lr 1e-4, gamma .99, gae .95, clip .2, n_steps 4*stage_steps, batch n_steps/10, epochs 5, ent .08 fresh / .01 resume, vf .3, net 256x256, vecnorm reward clip 10

reward (env_sb3)
- expl +1 new room
- guide +5 walk into room that still has an undelivered item while empty handed (resets after each delivery so cant farm)
- pickup +10 first time per item
- pen = carry shaping, potential based (prev_bfs-curr_bfs)*0.5 toward kitchen, oscillating cancels out
- delivery 100*(1+0.4*already delivered)
- completion +500 when stage done

THE BUG i found 5 jun - delivery farming
ep25400 only did 1/2 but delivery reward was ~285 (one delivery = 100??). turns out delivered ball just sat on the kitchen floor and other agents kept picking it up + dropping it again, each re-drop counted as a delivery and escalated the bonus, meanwhile the unique set stays at 1. so they just learned to huddle in the kitchen passing one banana around instead of getting the rest. fixed it - ball gets removed from grid on delivery now (env_multi). means ep25400 weights are junk, gotta retrain run 1 from scratch

runs - 4 total, ~1 day each, back to back
run 1 does triple duty (SQ1 ont arm + SQ3 shared arm + SQ5 reference) thats how this fits
1. full ont mappo, collect_food, prox 5  <- primary, redo after the fix
2. vanilla baseline, collect_food  <- needs the one code change, use_ontology=False (zero the ont state feats, kill the guide reward, keep everything else same so its a fair test). NOT the 3 little flags, those were SQ2
3. independent abox, collect_food, prox=0  <- no code, just the param
4. collect_trash full pipeline  <- SQ5

SQ5 note - old plan said zero shot load the checkpoint, cant do that, obs dims dont match (8 vs 16). so instead just swap GOAL_NAME=collect_trash, keep the whole ont pipeline, train fresh, show it still learns = framework generalises. compare curve vs run1 as a loose reference not a real baseline. if i really need a controlled version thats a 5th run (vanilla trash) only if theres time

metrics (same for all)
- items delivered per ep (main one)
- stage completion rate + which ep each stage gets reached = convergence speed
- avg50 reward curve
- items seen per ep
all already in checkpoints_ppo/training_log.csv, eval with evaluate_ppo.py

rough schedule
day 1-2  land the fix (done), add use_ontology flag, check both modes get past stage 1
day 3-6  run 1 (the scary one, watch the curriculum)
day 7-9  run 2 vanilla
day 10-12 run 3 prox=0
day 13-15 run 4 trash
day 16-21 plots + writing, maybe a 5th run if buffer

fallback (it wasnt even finishing the 2 item stage before the fix...)
if 8 items just wont converge -> cap curriculum at whatever deepest stage ALL conditions hit reliably (prob 4 items) and report everything at that same stage. clean 4 item comparison >> broken 8 item one. decide after run 1 then lock it

future work / dropped on purpose
- SQ2 component ablation (use_prior/use_connectivity/use_semantic + 2-3 runs)
- proper controlled SQ5 (vanilla vs ont on trash)
- real zero shot transfer needs item-count-invariant obs (per item attention / set encoder)

chapters: SQ1 = ch4, SQ3+SQ5 = ch5, SQ2 = future work
