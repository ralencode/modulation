use poloto::build;
use rustfft::num_complex::ComplexFloat;
use rustfft::{num_complex::Complex, FftPlanner};

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

fn plot_vector<T>(vector: &[T], timeline: &[f64], filename: &str, name: &str, x: &str, y: &str)
where
    T: Into<f64> + Copy,
{
    let svg = poloto::header();
    let dirname: &str = Path::new(filename).parent().unwrap().to_str().unwrap();
    fs::create_dir_all(&dirname).expect(&format!("Cannot create output directory: {}", dirname));

    fs::write(
        filename,
        poloto::frame()
            .with_tick_lines([true, true])
            .with_viewbox(svg.get_viewbox())
            .build()
            .data(
                poloto::plots!(build::plot(""))
                    .line(timeline.iter().zip(vector.into_iter().map(|&x| x.into()))),
            )
            .build_and_label((name, x, y))
            .append_to(poloto::header().dark_theme())
            .render_string()
            .expect("Could not build plot"),
    )
    .expect("Unable to save file result.svg");
}

fn create_timeline(disc_freq: usize, duration: f64) -> Vec<f64> {
    let samples_count = (duration * disc_freq as f64) as i32;
    let freq = 1.0 / disc_freq as f64;
    (0..samples_count).map(|i| i as f64 * freq as f64).collect()
}

fn harmonic(freq: f64, timeline: &Vec<f64>, amplitude: f64) -> Vec<f64> {
    timeline
        .iter()
        .map(|&t| amplitude * f64::sin(2.0 * PI * freq * t))
        .collect()
}

fn harmonic_to_meander(harmonic: &Vec<f64>) -> Vec<u8> {
    harmonic
        .iter()
        .map(|i| (if i > &0.0 { 1 } else { 0 }))
        .collect()
}

fn signal_spectrum<T>(signal: &Vec<T>) -> Vec<f64>
where
    T: Into<f64> + Copy,
{
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal.len());

    let mut buffer: Vec<Complex<f64>> = signal
        .into_iter()
        .map(|&x| Complex::new(x.into(), 0.0))
        .collect();
    fft.process(&mut buffer);

    buffer
        .iter()
        .take(signal.len() / 2)
        .map(|c| c.norm() / signal.len() as f64 * 2.0)
        .collect()
}

fn fft_complex<T>(signal: &Vec<T>) -> Vec<Complex<f64>>
where
    T: Into<f64> + Copy,
{
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal.len());

    let mut buffer: Vec<Complex<f64>> = signal
        .into_iter()
        .map(|&x| Complex::new(x.into(), 0.0))
        .collect();
    fft.process(&mut buffer);

    buffer
}

fn ifft_complex(spectrum: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(spectrum.len());
    let mut buffer = spectrum.clone();
    fft.process(&mut buffer);
    buffer.iter().map(|c| *c / spectrum.len() as f64).collect()
}

fn hilbert<T>(signal: &Vec<T>) -> Vec<f64>
where
    T: Into<f64> + Copy,
{
    let fft_result = fft_complex(signal);
    let len = fft_result.len();
    let mut hilbert_spectrum: Vec<Complex<f64>> = fft_result
        .into_iter()
        .enumerate()
        .map(|(i, c)| if i <= len / 2 { c } else { Complex::from(0.0) })
        .collect();

    hilbert_spectrum[0] = Complex::new(hilbert_spectrum[0].re / 2.0, 0.0);

    let ifft_result: Vec<Complex<f64>> = ifft_complex(&hilbert_spectrum);
    ifft_result.iter().map(|c| c.abs()).collect()
}

fn noise_signal(signal: &Vec<f64>) -> Vec<f64> {
    let n = signal.len();
    let one_percent = (n as f64 * 0.01).ceil() as usize;

    let mut buffer = fft_complex(&signal);

    for i in 0..one_percent {
        buffer[i] = Complex::new(0.0, 0.0);
        buffer[n - 1 - i] = Complex::new(0.0, 0.0);
    }
    buffer = buffer
        .iter()
        .map(|&i| {
            if i.abs() <= 0.3 {
                Complex::new(0.0, 0.0)
            } else {
                i
            }
        })
        .collect();
    ifft_complex(&buffer).iter().map(|c| c.re).collect()
}

fn amplitude_modulation<T>(signal: &Vec<T>, carrier_signal: &Vec<f64>) -> Vec<f64>
where
    T: Into<f64> + Copy,
{
    let m = 0.9;
    assert_eq!(signal.len(), carrier_signal.len());
    let max_signal = signal
        .into_iter()
        .fold(-f64::INFINITY, |a, &b| a.max(b.into()));
    signal
        .iter()
        .zip(carrier_signal.iter())
        .map(|pair| pair.1 * (1.0 + m * Into::<f64>::into(*pair.0) / max_signal))
        .collect()
}

fn amplitude_demodulation(modulated_signal: &Vec<f64>, carrier_signal: &[f64]) -> Vec<f64> {
    let m = 0.9;
    let epsilon = 1e-10;
    assert_eq!(modulated_signal.len(), carrier_signal.len());

    modulated_signal
        .iter()
        .zip(carrier_signal.iter())
        .map(|(&mod_signal, &carrier)| {
            if carrier.abs() < epsilon {
                0.0
            } else {
                (mod_signal / carrier - 1.0) / m
            }
        })
        .collect()
}

fn main() {
    let amplitude = 1.0;
    let duration = 5.0;
    let modulating_freq = 1.0;
    let carrier_freq = 10.0;
    let disc_freq = 900;

    let freqs: Vec<f64> = (0..disc_freq).map(|i| i as f64 * 1.0 / duration).collect();

    let timeline = create_timeline(disc_freq, duration);
    let carrier = harmonic(carrier_freq, &timeline, amplitude);
    let modulating_signal = harmonic_to_meander(&harmonic(modulating_freq, &timeline, amplitude));

    let modulated_signal = amplitude_modulation(&modulating_signal, &carrier);
    let modulated_spectrum = signal_spectrum(&modulated_signal);

    let dirname = String::from("out/modulating-and-carrier/");
    plot_vector(
        &modulating_signal,
        &timeline,
        &format!("{}{}.svg", dirname, "modulating-signal"),
        "Modulating signal",
        "time",
        "amplitude",
    );
    plot_vector(
        &carrier,
        &timeline,
        &format!("{}{}.svg", dirname, "carrier-signal"),
        "Carrier signal",
        "time",
        "amplitude",
    );

    let noised_signal = noise_signal(&modulated_signal);
    let demodulated_signal = amplitude_demodulation(&noised_signal, &carrier);
    let demodulated_signal: Vec<f64> = demodulated_signal
        .iter()
        .map(|value| if *value > 1.01 { 0.0 } else { *value })
        .map(|value| if value < -0.01 { 0.0 } else { value })
        .map(|value| if value > 0.8 { 1.0 } else { 0.0 })
        .collect();
    let demodulated_signal: Vec<f64> = demodulated_signal
        .iter()
        .take(demodulated_signal.len() - 51)
        .enumerate()
        .map(|(i, _)| {
            if demodulated_signal[i..i + 50]
                .iter()
                .filter(|&&num| num > 0.9)
                .count()
                > 1
            {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let noised_spectrum = signal_spectrum(&noised_signal);
    let hilberted_spectrum = hilbert(&noised_signal);
    let squared_spectrum: Vec<f64> = hilberted_spectrum
        .iter()
        .map(|&value| if value <= 0.0003 { 0.0 } else { 1.0 })
        .collect();

    let dirname = String::from("out/hilberts-ifft/");
    plot_vector(
        &modulated_signal,
        &timeline,
        &format!("{}{}.svg", dirname, "modulated-signal"),
        "Modulated signal",
        "time",
        "amplitude",
    );
    plot_vector(
        &noised_signal,
        &timeline,
        &format!("{}{}.svg", dirname, "noised-signal"),
        "Noised signal",
        "time",
        "amplitude",
    );
    plot_vector(
        &demodulated_signal,
        &timeline[0..demodulated_signal.len()],
        &format!("{}{}.svg", dirname, "demodulated-signal"),
        "Demodulated signal",
        "time",
        "amplitude",
    );
    plot_vector(
        &modulated_spectrum[0..100],
        &freqs[0..100],
        &format!("{}{}.svg", dirname, "modulated-spectrum"),
        "Modulated signal spectrum",
        "frequency",
        "amplitude",
    );
    plot_vector(
        &noised_spectrum[0..100],
        &freqs[0..100],
        &format!("{}{}.svg", dirname, "noised-spectrum"),
        "Noised signal spectrum",
        "frequency",
        "amplitude",
    );
    plot_vector(
        &hilberted_spectrum,
        &timeline[0..hilberted_spectrum.len()],
        &format!("{}{}.svg", dirname, "hilbert-spectrum"),
        "Hilbert spectrum",
        "frequency",
        "amplitude",
    );
    plot_vector(
        &squared_spectrum,
        &timeline[0..squared_spectrum.len()],
        &format!("{}{}.svg", dirname, "squared-spectrum"),
        "Squared Hilbert spectrum",
        "frequency",
        "amplitude",
    );

    let max_modulating_signal = modulating_signal
        .iter()
        .cloned()
        .fold(0.0 / 0.0, |a: f64, b| a.max(b as f64));
    let normalized_mod_signal: Vec<f64> = modulating_signal
        .iter()
        .map(|&amp| amp as f64 / max_modulating_signal)
        .collect();

    let freq_mod_signal: Vec<f64> = timeline
        .iter()
        .enumerate()
        .map(|(t, &time)| {
            let cumulative_sum: f64 = modulating_signal
                .iter()
                .take(t)
                .map(|&value| value as f64)
                .sum::<f64>()
                / disc_freq as f64;
            let modulation_term = cumulative_sum / max_modulating_signal * 2.0 * PI * carrier_freq;
            (2.0 * PI * carrier_freq * time + modulation_term).sin()
        })
        .collect();
    let freq_mod_spectrum = signal_spectrum(&freq_mod_signal);

    let phase_mod_signal: Vec<f64> = timeline
        .iter()
        .zip(normalized_mod_signal.iter())
        .map(|(&t, &mod_amp)| (2.0 * PI * carrier_freq * t + mod_amp).sin())
        .collect();
    let phase_mod_spectrum = signal_spectrum(&phase_mod_signal);

    let dirname = String::from("out/frequency-phase/");
    plot_vector(
        &freq_mod_signal,
        &timeline,
        &format!("{}{}.svg", dirname, "freq-modulated-signal"),
        "Frequency modulated signal",
        "time",
        "amplitude",
    );
    plot_vector(
        &freq_mod_spectrum[0..100],
        &freqs[0..100],
        &format!("{}{}.svg", dirname, "freq-modulated-spectrum"),
        "Frequency modulated spectrum",
        "frequency",
        "amplitude",
    );
    plot_vector(
        &phase_mod_signal,
        &timeline,
        &format!("{}{}.svg", dirname, "phase-modulated-signal"),
        "Phase modulated signal",
        "time",
        "amplitude",
    );
    plot_vector(
        &phase_mod_spectrum[0..100],
        &freqs[0..100],
        &format!("{}{}.svg", dirname, "phase-modulated-spectrum"),
        "Phase modulated spectrum",
        "frequency",
        "amplitude",
    );
}
