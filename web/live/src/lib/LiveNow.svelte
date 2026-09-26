<script lang="ts">
	// The games being played now, one card each, like a streaming site's live channels:
	// the PC's screen a few seconds ago, how long it has run, the map and side, and which
	// PC plays it. The one on the player above says so; another tapped goes up there.
	import { SIDE, arena, clock, hhmm } from './format';
	import type { Station } from './types';

	let {
		stations,
		watching,
		onpick
	}: { stations: Station[]; watching: string | null; onpick: (id: string) => void } = $props();

	// The screens are asked for again every 10 s.
	let tick = $state(0);
	$effect(() => {
		const timer = setInterval(() => tick++, 10000);
		return () => clearInterval(timer);
	});
</script>

<ul>
	{#each stations as s (s.id)}
		{@const game = s.game}
		{#if game}
			<li>
				<button class:on={s.id === watching} onclick={() => onpick(s.id)}>
					<div class="thumb">
						<img
							src={`s/${s.id}/latest.jpg?${tick}`}
							alt=""
							onload={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = '')}
							onerror={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = 'hidden')}
						/>
						<span class="live">LIVE</span>
						<span class="time">{clock(game.elapsed)}</span>
						{#if s.id === watching}<span class="watching">Watching</span>{/if}
					</div>
					<div class="title">
						{arena(game.arena)} <span class={game.side}>{SIDE[game.side] ?? game.side}</span>
					</div>
					<div class="muted">
						{[s.label, game.plan?.variant ? `${game.plan.variant} plan` : '', `since ${hhmm(game.started_unix)}`]
							.filter(Boolean)
							.join(' · ')}
					</div>
				</button>
			</li>
		{/if}
	{/each}
</ul>

<style>
	ul {
		list-style: none;
		margin: 0;
		padding: 0 16px;
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
		gap: 14px;
	}
	button {
		display: block;
		width: 100%;
		padding: 0;
		background: none;
		border: 0;
		text-align: left;
		cursor: pointer;
	}
	.thumb {
		position: relative;
		aspect-ratio: 16 / 9;
		background: #000;
		border-radius: 10px;
		overflow: hidden;
		border: 2px solid transparent;
	}
	button.on .thumb {
		border-color: var(--live);
	}
	img {
		width: 100%;
		height: 100%;
		object-fit: cover;
		display: block;
	}
	.live,
	.time,
	.watching {
		position: absolute;
		font-size: 12px;
		font-weight: 700;
		padding: 1px 7px;
		border-radius: 5px;
	}
	.live {
		top: 8px;
		left: 8px;
		background: var(--live);
		color: #fff;
		letter-spacing: 0.5px;
	}
	.time {
		bottom: 8px;
		right: 8px;
		background: rgba(0, 0, 0, 0.75);
		font-variant-numeric: tabular-nums;
	}
	.watching {
		bottom: 8px;
		left: 8px;
		background: rgba(0, 0, 0, 0.75);
	}
	.title {
		margin-top: 6px;
		font-weight: 650;
	}
	.muted {
		font-size: 13px;
	}
</style>
