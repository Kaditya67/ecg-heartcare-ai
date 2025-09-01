from django.core.management.base import BaseCommand
from api.models import ECGLabel

class Command(BaseCommand):
    help = "Seed or update ECG abnormality labels with enriched descriptions"

    def handle(self, *args, **kwargs):
        labels = [
            (1, "PAC", "#1f77b4",
             "Premature Atrial Contractions (PACs)"
             "Clinical: Early atrial beats that disrupt normal rhythm. "
             "Pattern: Normal–PAC–Normal sequence. "
             "Waveform: Abnormal or early P wave before a normal QRS."),
            
            (2, "PVC", "#d62728",
             "Premature Ventricular Contractions (PVCs)"
             "Clinical: Premature depolarization from ventricles, may be benign or pathologic. "
             "Pattern: Wide QRS interrupting sinus rhythm, may occur isolated or in repeating forms. "
             "Waveform: No preceding P wave, QRS is wide (>120ms), T wave often opposite in polarity."),
            
            (3, "Sinus Bradycardia", "#2ca02c",
             "Clinical: Normal sinus rhythm with slow rate (<60 bpm). "
             "Pattern: Regular but widely spaced beats. "
             "Waveform: Normal P–QRS–T sequence, but long RR intervals."),
            
            (4, "Sinus Tachycardia", "#ff7f0e",
             "Clinical: Normal sinus rhythm with fast rate (>100 bpm). "
             "Pattern: Regular, closely spaced beats. "
             "Waveform: Normal P–QRS–T, but shortened RR interval."),
            
            (5, "AFib", "#9467bd",
             "Atrial fibrillation (AFib)"
             "Clinical: Disorganized atrial activity, risk of stroke. "
             "Pattern: Irregularly irregular QRS spacing. "
             "Waveform: No distinct P waves, fibrillatory baseline between QRS."),
            
            (6, "Atrial Flutter", "#e377c2",
             "Clinical: Rapid atrial depolarizations (~250–350 bpm). "
             "Pattern: Sawtooth flutter waves, often with 2:1 or 3:1 conduction. "
             "Waveform: Multiple flutter (F) waves replacing normal P waves."),
            
            (7, "VTach", "#8c564b",
             "Ventricular Tachycardia (VTach)"
             "Clinical: Dangerous rhythm, can progress to VFib. "
             "Pattern: ≥3 PVCs in a row, rapid and regular. "
             "Waveform: No P waves, wide and tall QRS complexes."),
            
            (8, "VFib", "#7f7f7f",
             "Ventricular Fibrillation (VFib)"
             "Clinical: Cardiac arrest rhythm, no effective output. "
             "Pattern: Completely chaotic baseline. "
             "Waveform: No identifiable P, QRS, or T waves."),
            
            (9, "Ventricular Ectopic", "#bcbd22",
             "Clinical: Single premature ventricular beat. "
             "Pattern: Isolated PVC within sinus rhythm. "
             "Waveform: Wide QRS without preceding P wave."),
            
            (10, "Couplets", "#17becf",
             "Clinical: Increased risk for VTach. "
             "Pattern: Two PVCs back-to-back (PVC–PVC). "
             "Waveform: Wide consecutive QRS without P waves."),
            
            (11, "Triplets", "#00bfff",
             "Clinical: Short run of VTach. "
             "Pattern: Three PVCs in a row (PVC–PVC–PVC). "
             "Waveform: Wide, tall QRS, no P waves, resembles VTach."),
            
            (12, "PVC Subtypes", "#ffd700",
             "Clinical: Variations of PVC occurrence. "
             "Pattern: Bigeminy (Normal–PVC), Trigeminy (Normal–Normal–PVC), Quadrigeminy (Normal–Normal–Normal–PVC), "
             "Unifocal (all PVCs same), Multifocal (PVCs differ). "
             "Waveform: Wide QRS, no P before PVC, morphology depends on origin."),
        ]

        for value, name, color, description in labels:
            obj, created = ECGLabel.objects.update_or_create(
                value=value,
                defaults={
                    "name": name,
                    "color": color,
                    "description": description
                }
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f"Added {name}"))
            else:
                self.stdout.write(self.style.WARNING(f"Updated {name}"))
