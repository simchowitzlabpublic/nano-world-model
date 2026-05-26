// ─── Site Content ─────────────────────────────────────────────────────────────
// Edit this file to update all page text. No HTML knowledge needed.
// After saving, refresh the browser to see changes.

const CONTENT = {

    hero: {
        title: 'Nano World Models',
        subtitle: 'A minimalist, batteries-included repository for advancing world model science',
        links: {
            arxiv: 'https://arxiv.org/abs/2605.23993',
            github: 'https://github.com/simchowitzlabpublic/nano-world-model',
            models: 'https://huggingface.co/collections/knightnemo/nano-world-model',
            twitter: 'https://twitter.com',
        },
    },

    motivation: {
        label: '01 — Motivation',
        title: 'From Models to Science',
        body: [
            `World Models have emerged as a <strong>cornerstone of spatial intelligence and real-world decision-making</strong>, generating high-fidelity futures by conditioning on the agent's history and actions. Yet, for the broader community, the gap between <em>reading</em> about these models and <em>deploying</em> them remains disappointingly wide.`,
            `Research in this area is paradoxically <strong>fragmented yet converging</strong>. Fragmented, because implementations are often siloed, tied to specific datasets, and support few downstream tasks (e.g., policy evaluation, planning, MBRL). Converging, because the community is aligning on a consistent family of algorithms centered around streaming video generation, especially <em>diffusion-forcing methods</em>.`,
            `As the algorithm stabilizes, a shift in research focus lies from <em>inventing new techniques</em> to understanding the <strong>Science of World Models</strong>. Two forces drive this shift:<span class="motivation-forces"><span>(a.) The <strong>computation bar</strong> for developing techniques that may actually prove to be useful at industry-scale models keeps rising, and</span><span>(b) <strong>Algorithmic convergence</strong> makes insights about model behavior broadly applicable.</span></span>Yet the Science of World Models (e.g. effects of pretraining, scaling laws, and careful studies of empirical best practices), remains largely underexplored.`,
            `<strong>Nano World Models</strong> is built to close this gap. We argue that one cannot truly understand the science behind a model without being able to easily experiment with it. This repository offers a <em>minimalist, batteries-included</em> implementation of diffusion forcing based world models, engineered to support diverse data formats, design choices and downstream applications out-of-the-box. By prioritizing clean abstractions and comprehensive documentation, this repo can be modded, adapted, and built upon with minimal overhead. We hope this repo provides a useful toolkit for world modeling, aiding science of world model research as well as promoting world model accessibility for the broader community.`,
        ],
        quotes: [
            '',
            '<span class="page-quote__text">The eternal mystery of the world is its comprehensibility... The fact that it is comprehensible is a miracle.</span><span class="page-quote__author">Albert Einstein</span>',
            '',
        ],
        quote: '',
    },

    quickstart: {
        label: '02 — Quick-start',
        title: 'Get Started',
        intro: `Thanks to its <strong>modular design</strong>, Nano World Models makes experimentation a matter of changing <em>modular configs</em> rather than rewriting pipelines. Datasets, tasks, model sizes, prediction objectives and overriding any specific configuration, can be swapped from a <strong>single command line</strong>.`,
        code: `python src/main.py \\
  dataset=dino_wm/pusht \\  # data: dino_wm/* | lerobot/* | rt1/* | game/csgo
  experiment=train \\  # task: train | evaluate_only | planning
  model=nanowm_b2 \\  # size: nanowm_s2 (30M) | nanowm_b2 | nanowm_l2 (500M+)
  diffusion.pred_name=v \\  # objective: v | eps | x
  model.action_injection.type=adaln \\  # action: additive | adaln | film | cross_attention
  infra.num_nodes=1 \\  # scale: 1 GPU → 32 GPUs, no pipeline changes`,
        note: '',
    },

    architecture: {
        label: '03 — Architecture',
        title: 'Modular by design',
        body: 'A shallow map of the codebase: one entrypoint, composable configs, and small modules that can be swapped or inspected independently.',
        tree: [
            { name: 'configs/', desc: 'Hydra config groups' },
            { name: 'wm_datasets/', desc: 'data adapters' },
            { name: 'models/', desc: 'NanoWM backbone' },
            { name: 'diffusion/', desc: 'sampling objectives' },
            { name: 'experiments/', desc: 'train / eval' },
            { name: 'planning/', desc: 'CEM + MPC' },
            { name: 'viewer/', desc: 'interactive demos' },
            { name: 'main.py', desc: 'Hydra launch' },
            { name: '...', desc: 'tools, scripts, utils' },
        ],
        modules: [
            {
                num: '01',
                title: '`wm_datasets/`',
                desc: 'Data loading',
            },
            {
                num: '02',
                title: '`models/`',
                desc: 'Backbone + action injection',
            },
            {
                num: '03',
                title: '`diffusion/`',
                desc: 'v / eps / x objectives',
            },
            {
                num: '04',
                title: '`experiments/`',
                desc: 'Train + eval orchestration',
            },
            {
                num: '05',
                title: '`planning/`',
                desc: 'CEM + MPC loop',
            },
            {
                num: '06',
                title: '`viewer/`',
                desc: 'Interactive visualization demo',
            },
        ],
    },

    features: {
        label: '04 — Features',
        title: 'Support for the full stack',
        interactionHint: 'Click any card for details &rarr;',
        cards: [
            {
                num: '01',
                title: 'Diverse Environments',
                body: 'DINO-WM toy datasets (point maze, push-T, rope, granular), LeRobot-style real-world datasets, RT-1, and complex video datasets (CSGO). One config line to swap.',
                detailImage: {
                    src: 'assets/dataset_overview.png',
                    alt: 'DINO-WM, LeRobot RT-1, and CSGO dataset examples',
                },
            },
            {
                num: '02',
                title: 'Prediction Objectives',
                body: 'x-prediction, ε-prediction, and v-prediction — all supported via <code>diffusion.pred_name</code>. Ablation scripts included for direct comparison.',
                details: [
                    'Switch between x, epsilon, and velocity prediction with <code>diffusion.pred_name</code>.',
                    'Use the same model and data plumbing while isolating the objective as the variable under study.',
                    'Ablation scripts are included so objective comparisons are reproducible rather than one-off edits.',
                ],
            },
            {
                num: '03',
                title: 'Action Injection Methods',
                body: 'Five injection strategies: additive, AdaLN, AdaLN-fuse, FiLM, cross-attention. Configurable via <code>model.action_injection.type</code>.',
                detailImage: {
                    src: 'assets/action_injection.png',
                    alt: 'Action injection method comparison: additive, AdaLN, AdaLN-fuse, FiLM, cross-attention',
                },
            },
            {
                num: '04',
                title: 'Scale',
                body: 'Nano-WM can be trained on from a single GPU to multiple nodes of GPUs (e.g. 4 nodes with 8 GPUs each), with model sizes ranging from 30M to 500M+.',
                detailTable: {
                    caption: 'Ablation: Model Scale (RT-1, 50K steps)',
                    headers: ['Architecture', 'Params', 'PSNR &uarr;', 'SSIM &uarr;', 'LPIPS &darr;', 'FID &darr;'],
                    rows: [
                        ['NanoWM-S/2', '39.8M', '22.30', '0.739', '0.230', '54.95'],
                        ['NanoWM-B/2', '158.6M', '23.07', '0.760', '0.207', '42.27'],
                        ['<strong>NanoWM-L/2</strong>', '~460M', '<strong>23.62</strong>', '<strong>0.777</strong>', '<strong>0.186</strong>', '<strong>36.31</strong>'],
                    ],
                    highlightRow: 2,
                },
            },
            {
                num: '05',
                title: 'Logging & Evaluation',
                body: 'Detailed WandB + TensorBoard logging, callback-style validation, reliable per-step checkpointing. Evaluation is fixed-seed reproducible.',
                detailImage: {
                    src: 'assets/wandb.png',
                    alt: 'WandB dashboard showing training and validation curves for Nano-WM runs',
                },
                detailLink: {
                    label: 'View live run on WandB &rarr;',
                    href: 'https://wandb.ai/better_guidance/nano-world-model-phase2/runs/lho54iwn?nw=nwuserhuangsq23',
                },
            },
        ],
    },

    applications: {
        label: '05 — Applications',
        title: 'Downstream applications',
        interactionHint: 'Click any card for details &rarr;',
        cards: [
            {
                icon: '&#9650;',
                title: 'Long-horizon Rollout',
                body: 'Autoregressive rollout with diffusion forcing for temporally consistent long video generation',
                detailVideo: {
                    src: 'assets/csgo_100k_long_rollout.mp4',
                    alt: 'Long-horizon CSGO rollout from a 100K-step Nano-WM checkpoint',
                    notCherryPicked: true,
                },
            },
            {
                icon: '&#9632;',
                title: 'Video &rarr; 3D Map',
                body: 'Long-horizon rollouts converted to 3D point cloud maps via frontier 3D reconstruction models (e.g. <a href="https://depth-anything-3.github.io/" target="_blank" rel="noopener noreferrer">DA3</a>).',
                detailVideo: {
                    src: 'assets/video_to_3d.mp4',
                    alt: 'Nano-WM rollout lifted into a 3D point cloud map via Depth-Anything-3',
                    notCherryPicked: true,
                },
            },
            {
                icon: '&#9679;',
                title: 'MPC-style Planning',
                body: 'CEM-style MPC over the diffusion world model: sample action sequences, roll them out in VAE latent space, and update the sampling distribution toward the lowest-MSE elites against the goal latent.',
                details: [
                    'Each replan samples action sequences, rolls them through the world model in VAE latent space, scores them by MSE to the goal latent, and refits the proposal distribution to the top-k elites.',
                    'Closed-loop control with configurable horizon, replan cadence, CEM sampling budget, and episode length — drop in any compatible environment to evaluate.',
                    'Same checkpoint that generates rollouts also plans actions — no separate policy network, no fine-tuning.',
                ],
            },
        ],
    },

    cta: {
        label: '06 — Open Source',
        title: 'World Models Need the World',
        body: 'Our mission is to build a <strong>Babel Tower for world-model research</strong>: a shared, open stack where datasets, objectives, architectures, and downstream tasks can speak the same language. Through <strong>reproducible experiments</strong>, <strong>modular design</strong>, and <strong>standardized benchmarks</strong>, we hope our work lays a solid foundation for understanding the <em>science of world models</em>. Ultimately, <em>World Models need the world</em>, and we invite the global community to join us, contribute, and build this future together.',
        links: {
            arxiv: 'https://arxiv.org/abs/2605.23993',
            github: 'https://github.com/simchowitzlabpublic/nano-world-model',
            models: 'https://huggingface.co/collections/knightnemo/nano-world-model',
        },
        citation: `@misc{huang2026nanoworldmodels,
  title={Nano World Models: A Minimalist Implementation of Future Video Prediction},
  author={Siqiao Huang and Partha Kaushik and Michael Chen and Hengkai Pan and Omar Chehab and Fernando Moreno-Pino and Max Simchowitz},
  year={2026},
  eprint={2605.23993},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.23993},
}`,
        copyright: '© 2026 All rights reserved',
    },

};
