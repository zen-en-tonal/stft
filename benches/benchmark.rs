use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stft::{Stft, Q};

fn stft(q: Q, num_sample: usize) {
    let mut stft: Stft<f32> = Stft::with_hann(q);
    let mut samples = vec![0.; num_sample];
    stft.process(&mut samples, |_| {});
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("stft window=128 overrap=0.75 44100 samples", |b| {
        b.iter(|| stft(black_box(Q::new(7, 0.75)), black_box(44100)))
    });
    c.bench_function("stft window=128 overrap=0.5 44100 samples", |b| {
        b.iter(|| stft(black_box(Q::new(7, 0.5)), black_box(44100)))
    });
    c.bench_function("stft window=128 overrap=0.25 44100 samples", |b| {
        b.iter(|| stft(black_box(Q::new(7, 0.25)), black_box(44100)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
