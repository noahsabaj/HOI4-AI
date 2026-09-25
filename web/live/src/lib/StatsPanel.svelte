<script lang="ts">
	// The record by map and plan over the last two days, and training in progress.
	import { get } from './api';
	import { ago, arena } from './format';
	import type { Stats, Tally } from './types';

	let stats: Stats | null = $state(null);
	const plans = ['best', 'tuned', 'explore', 'learned'];

	get<Stats>('api/stats').then((s) => (stats = s));

	const games = (c: Tally) => c.win + c.loss + c.timeout;
</script>

{#if stats}
	<h3>Last {stats.hours} hours, wins by map</h3>
	<table>
		<thead><tr><th>Map</th>{#each plans as plan (plan)}<th>{plan}</th>{/each}</tr></thead>
		<tbody>
			{#each Object.entries(stats.by_arena).sort() as [name, row] (name)}
				<tr>
					<td>{arena(name)}</td>
					{#each plans as plan (plan)}
						{@const c = row[plan]}
						{#if c}
							<td>{c.win}/{games(c)} <span class="muted">{Math.round((100 * c.win) / games(c))}%</span></td>
						{:else}
							<td class="muted">–</td>
						{/if}
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
	{#if stats.training.length}
		<h3>Training</h3>
		<table>
			<thead><tr><th>Run</th><th>Epoch</th><th>Step</th><th>Loss</th><th>Updated</th></tr></thead>
			<tbody>
				{#each stats.training as t (t.run)}
					<tr><td>{t.run}</td><td>{t.epoch}</td><td>{t.step}</td><td>{t.loss}</td><td>{ago(t.updated_unix)}</td></tr>
				{/each}
			</tbody>
		</table>
	{/if}
{:else}
	<p class="muted">Loading…</p>
{/if}

<style>
	table {
		border-collapse: collapse;
		width: 100%;
		font-size: 13px;
	}
	th,
	td {
		text-align: left;
		padding: 4px 6px;
		border-bottom: 1px solid var(--line);
	}
	h3 {
		font-size: 14px;
		margin: 14px 0 6px;
	}
</style>
