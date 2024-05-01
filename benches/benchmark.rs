use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stft::Stft;

fn stft(level: usize, hop_size: usize, num_sample: usize) {
    let mut stft: Stft<f32> = Stft::with_hann(level, hop_size);
    let mut samples = vec![0.; num_sample];
    stft.process(&mut samples, |_| {});
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("stft window=128 hop_size=64 44100 samples", |b| {
        b.iter(|| stft(black_box(7), black_box(64), black_box(44100)))
    });
    c.bench_function("stft window=128 hop_size=32 44100 samples", |b| {
        b.iter(|| stft(black_box(7), black_box(32), black_box(44100)))
    });
    c.bench_function("stft window=128 hop_size=16 44100 samples", |b| {
        b.iter(|| stft(black_box(7), black_box(16), black_box(44100)))
    });
    c.bench_function("stft window=128 hop_size=8 44100 samples", |b| {
        b.iter(|| stft(black_box(7), black_box(8), black_box(44100)))
    });
    c.bench_function("stft window=128 hop_size=4 44100 samples", |b| {
        b.iter(|| stft(black_box(7), black_box(4), black_box(44100)))
    });
    c.bench_function("stft window=128 hop_size=2 44100 samples", |b| {
        b.iter(|| stft(black_box(7), black_box(2), black_box(44100)))
    });
    c.bench_function("stft window=128 hop_size=1 44100 samples", |b| {
        b.iter(|| stft(black_box(7), black_box(1), black_box(44100)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
