use poloto::build;
use rustfft::{FftPlanner, num_complex::Complex};

use std::fmt::Display;
use std::fs;
use std::f64::consts::PI;

fn plot_vector<T>(vector: &Vec<T>, timeline: &Vec<f64>, name: &str, x: &str, y: &str) -> String
where
    T: Into<f64> + Copy
{
    let svg = poloto::header();

    poloto::frame()
        .with_tick_lines([true, true])
        .with_viewbox(svg.get_viewbox())
        .build()
        .data(poloto::plots!(build::plot("")).line(timeline.iter().zip(vector.into_iter().map(|&x| x.into()))))
        .build_and_label((name, x, y))
        .append_to(poloto::header().dark_theme())
        .render_string().expect("Could not build plot")
}

fn create_timeline(samples_per_second: usize, duration: f64) -> Vec<f64> {
    let samples_count = (duration * samples_per_second as f64) as i32;
    let freq = 1.0 / samples_per_second as f64;
    (0..samples_count).map(|i| i as f64 * freq as f64).collect()
}

fn harmonic(freq: f64, timeline: &Vec<f64>) -> Vec<f64> {
    timeline.iter().map(|&t| f64::sin(2.0 * PI * freq * t)).collect()
}

fn harmonic_to_meander(wave: &Vec<f64>) -> Vec<u8> {
    wave.iter().map(|i| (if i > &0.0 { 1 } else { 0 })).collect()
}

fn signal_spector<T>(signal: &Vec<T>) -> Vec<f64> 
where
    T: Into<f64> + Copy + Display
{

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(signal.len());

    let mut buffer: Vec<Complex<f64>> = signal
        .into_iter()
        .map(|&x| Complex::new(x.into(), 0.0))
        .collect();
    fft.process(&mut buffer);

    buffer.iter().take(signal.len() / 2).map(|c| c.norm()).collect()
}

fn main() {
    let disc_freq = 500;
    let duration = 16.0;
    let freqs = [1.0, 2.0, 4.0, 8.0];
    for &freq in freqs.iter() {
        let suffix = format!("{}", freq as u8);
        let ext = ".svg";

        let dirname = &format!("out/freq{}/", suffix)[..];

        fs::create_dir_all(dirname).expect("Cannot create output directory");
        
        let harmonic_name = &format!("harmonic-freq{}", suffix)[..];
        let meander_name = &format!("meander-freq{}", suffix)[..];

        let duration = duration / freq;
        let timeline = create_timeline(disc_freq, duration);

        let length = timeline.len();
        let amplitudes = (0..length / 2).map(|i| i as f64 * disc_freq as f64 / length as f64).collect();
    
        let sine_graph = harmonic(freq, &timeline);
        let square_graph = harmonic_to_meander(&sine_graph);
        
        fs::write(format!("{}{}{}", dirname, harmonic_name, ext), plot_vector(&sine_graph, &timeline, "harmonic signal", "time", "amplitude")).expect("Unable to save file result.svg");
        fs::write(format!("{}{}{}", dirname, meander_name, ext), plot_vector(&square_graph, &timeline, "meander signal", "time", "amplitude")).expect("Unable to save file result.svg");

        let sine_spector = signal_spector(&sine_graph);
        let square_spector = signal_spector(&square_graph);

        fs::write(format!("{}{}-spector{}", dirname, harmonic_name, ext), plot_vector(&sine_spector, &amplitudes, "spector of harmonic signal", "frequency", "amplitude")).expect("Unable to save file result.svg");
        fs::write(format!("{}{}-spector{}", dirname, meander_name, ext), plot_vector(&square_spector, &amplitudes, "spector of meander signal", "frequency", "amplitude")).expect("Unable to save file result.svg");
    }
}
