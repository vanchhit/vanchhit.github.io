export type CoolItem = {
  title: string;
  url?: string;
  category: string;
  excerpt: string;
};

export const coolStuff: CoolItem[] = [
  // Tools
  {
    title: 'FlightRadar24',
    url: 'https://www.flightradar24.com/',
    category: 'Tools',
    excerpt: 'The obvious one. Still the best window onto live air traffic.',
  },
  {
    title: 'Planespotters.net',
    url: 'https://www.planespotters.net',
    category: 'Tools',
    excerpt: 'Aircraft registries, fleet lists, spotting photos from airports worldwide.',
  },
  {
    title: 'Seat61',
    url: 'https://www.seat61.com',
    category: 'Tools',
    excerpt: "The definitive guide to travelling by train. Mark Smith's life work.",
  },
  {
    title: 'Rome2rio',
    url: 'https://www.rome2rio.com',
    category: 'Tools',
    excerpt: 'Multi-modal journey planner. Great for figuring out overland routes.',
  },

  // Data
  {
    title: 'Our World in Data — Energy',
    url: 'https://ourworldindata.org/energy',
    category: 'Data',
    excerpt: 'Charts you can actually trust, with sources.',
  },
  {
    title: 'Ember Climate',
    url: 'https://ember-climate.org',
    category: 'Data',
    excerpt: 'Independent energy think-tank with clean, open data visualisations.',
  },
  {
    title: 'Global Carbon Budget',
    url: 'https://www.globalcarbonproject.org/carbonbudget/',
    category: 'Data',
    excerpt: 'Annual accounting of global CO₂ emissions and remaining carbon budget.',
  },

  // Reading
  {
    title: 'Marginalia by John Naughton',
    url: 'https://memex.naughtons.org/',
    category: 'Reading',
    excerpt: 'A long-running, generous-minded weblog. Worth subscribing to.',
  },
  {
    title: 'Alignment Forum',
    url: 'https://www.alignmentforum.org',
    category: 'Reading',
    excerpt: 'The best place to follow technical AI safety research.',
  },
  {
    title: '80,000 Hours',
    url: 'https://80000hours.org',
    category: 'Reading',
    excerpt: 'Research-backed career advice focused on doing the most good.',
  },
  {
    title: 'Works in Progress',
    url: 'https://worksinprogress.co',
    category: 'Reading',
    excerpt: 'Essays on progress, science, and what actually moves the world forward.',
  },
];
