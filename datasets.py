# Author: Kyle Kastner
# License: BSD 3-clause
# pulled from pthbldr
# Ideas from Junyoung Chung and Kyunghyun Cho
# See https://github.com/jych/cle for a library in this style
from music21 import converter, interval, pitch, harmony, analysis, spanner, midi, meter
import numpy as np
from collections import Counter
from scipy.io import loadmat, wavfile
from scipy.linalg import svd
from functools import reduce
import shutil
import string
import tarfile
import fnmatch
import zipfile
import gzip
import os
import json
import re
import csv
import time
import signal
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle

floatX = "float32"

def get_dataset_dir(dataset_name, data_dir=None, folder=None, create_dir=True):
    pth = os.getcwd() + os.sep + dataset_name
    if not os.path.exists(pth):
        os.mkdir(pth)
    return pth


def download(url, server_fname, local_fname=None, progress_update_percentage=5):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    try:
        import urllib
        urllib.urlretrieve('http://google.com')
    except AttributeError:
        import urllib.request as urllib
    u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def music21_to_chord_duration(p):
    """
    Takes in a Music21 score, and outputs two lists
    List for chords (by string name)
    List for durations
    """
    p_chords = p.chordify()
    p_chords_o = p_chords.flat.getElementsByClass('Chord')
    chord_list = []
    duration_list = []
    for ch in p_chords_o:
        chord_list.append(ch.primeFormString)
        #chord_list.append(ch.pitchedCommonName)
        duration_list.append(ch.duration.quarterLength)
    return chord_list, duration_list


def music21_to_pitch_duration(p):
    """
    Takes in a Music21 score, outputs 3 list of list
    One for pitch
    One for duration
    list for part times of each voice
    """
    parts = []
    parts_times = []
    parts_delta_times = []
    for i, pi in enumerate(p.parts):
        part = []
        part_time = []
        part_delta_time = []
        total_time = 0
        for n in pi.stream().flat.notesAndRests:
            if n.isRest:
                part.append(0)
            else:
                try:
                    part.append(n.midi)
                except AttributeError:
                    continue
            part_time.append(total_time + n.duration.quarterLength)
            total_time = part_time[-1]
            part_delta_time.append(n.duration.quarterLength)
        parts.append(part)
        parts_times.append(part_time)
        parts_delta_times.append(part_delta_time)
    return parts, parts_times, parts_delta_times


# http://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
# only works on Unix platforms though
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise ValueError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def _single_extract_music21(files, data_path, skip_chords, verbose, force_time_sig_denom, n):
    if verbose:
        print("Starting file {} of {}".format(n, len(files)))
    f = files[n]
    file_path = os.path.join(data_path, f)

    start_time = time.time()

    try:
        p = converter.parse(file_path)
        k = p.analyze("key")
        parse_time = time.time()
        if verbose:
            r = parse_time - start_time
            print("Parse time {}:{}".format(f, r))
    except (AttributeError, IndexError, UnicodeDecodeError,
            UnicodeEncodeError, harmony.ChordStepModificationException,
            ZeroDivisionError,
            ValueError,
            midi.MidiException,
            analysis.discrete.DiscreteAnalysisException,
            pitch.PitchException,
            spanner.SpannerException) as err:
        print("Parse failed for {}".format(f))
        return ("null",)

    p.keySignature = k

    # none if there is no data aug
    an = "B" if "major" in k.name else "D"

    try:
        time_sigs = [str(ts).split(" ")[-1].split(">")[0] for ts in p.recurse().getElementsByClass(meter.TimeSignature)]
        nums = [int(ts.split("/")[0]) for ts in time_sigs]
        num_check = all([n == nums[0] for n in nums])
        denoms = [int(ts.split("/")[1]) for ts in time_sigs]
        denom_check = all([d == denoms[0] for d in denoms])
        if force_time_sig_denom is not None:
            quarter_check = denoms[0] == force_time_sig_denom
        else:
            quarter_check = True
        if not num_check or not denom_check or not quarter_check:
            raise TypeError("Invalid")

        pc = pitch.Pitch(an)
        i = interval.Interval(k.tonic, pc)
        p = p.transpose(i)
        k = p.analyze("key")
        transpose_time = time.time()
        if verbose:
            r = transpose_time - start_time
            print("Transpose time {}:{}".format(f, r))

        if skip_chords:
            chords = ["null"]
            chord_durations = ["null"]
        else:
            chords, chord_durations = music21_to_chord_duration(p)
        pitches, parts_times, parts_delta_times = music21_to_pitch_duration(p)
        pitch_duration_time = time.time()
        if verbose:
            r = pitch_duration_time - start_time
            print("music21 to pitch_duration time {}:{}".format(f, r))
    except TypeError:
        #raise ValueError("Non-transpose not yet supported")
        return ("null",)
        """
        pc = pitch.Pitch(an)
        i = interval.Interval(k.tonic, pc)
        # FIXME: In this case chords are unnormed?
        if skip_chords:
            chords = ["null"]
            chord_durations = ["null"]
        else:
            chords, chord_durations = music21_to_chord_duration(p)
        pitches, durations = music21_to_pitch_duration(p)

        kt = k.tonic.pitchClass
        pct = pc.pitchClass
        assert kt >= 0
        if kt <= 6:
            pitches -= kt
        else:
            pitches -= 12
            pitches += (12 - kt)
        # now centered at C

        if "minor" in k.name:
            # C -> B -> B flat -> A
            pitches -= 3

        if pct <= 6:
            pitches += pct
        else:
            pitches -= 12
            pitches += pct
        """

    str_key = "{} minor".format(an) if "minor" in k.name else "{} major".format(an)

    ttime = time.time()
    if verbose:
        r = ttime - start_time
        print("Overall file time {}:{}".format(f, r))
    str_time_sig = time_sigs[0]
    return (pitches, parts_times, parts_delta_times, str_key, str_time_sig, f, p.quarterLength, chords, chord_durations)


# http://stackoverflow.com/questions/29494001/how-can-i-abort-a-task-in-a-multiprocessing-pool-after-a-timeout
def abortable_worker(func, *args, **kwargs):
    # returns ("null",) if timeout
    timeout = kwargs.get('timeout', None)
    p = multiprocessing.dummy.Pool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        return ("null",)


def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


def _music_extract(data_path, pickle_path, ext=".xml",
                   pitch_augmentation=False,
                   skip_chords=True,
                   skip_drums=True,
                   lower_voice_limit=None,
                   upper_voice_limit=None,
                   equal_voice_count=4,
                   force_denom=None,
                   parse_timeout=100,
                   multiprocess_count=4,
                   verbose=False):

    if not os.path.exists(pickle_path):
        print("Pickled file %s not found, creating. This may take a few minutes..." % pickle_path)
        itime = time.time()

        all_transposed_pitch = []
        all_transposed_parts_times = []
        all_transposed_parts_delta_times = []
        all_transposed_keys = []
        all_time_sigs = []
        all_file_names = []
        all_transposed_chord = []
        all_transposed_chord_duration = []
        all_quarter_length = []

        if 'basestring' not in globals():
            basestring = str

        if isinstance(data_path, basestring):
            files = sorted([fi for fi in os.listdir(data_path) if fi.endswith(ext)])
        else:
            files = sorted([ap for ap in data_path if ap.endswith(ext)])

        #import pretty_midi
        print("Processing {} files".format(len(files)))
        force_denom = 4
        if multiprocess_count is not None:
            from multiprocessing import Pool
            import functools
            pool = Pool(4)

            ex = functools.partial(_single_extract_music21,
                                   files, data_path,
                                   skip_chords, verbose, force_denom)
            abortable_ex = functools.partial(abortable_worker, ex, timeout=parse_timeout)
            result = pool.map(abortable_ex, range(len(files)))
            pool.close()
            pool.join()
        else:
            result = []
            for n in range(len(files)):
                r = _single_extract_music21(files, data_path, skip_chords,
                                            verbose, force_denom, n)
                result.append(r)

        for n, r in enumerate(result):
            if r[0] != "null":
                (pitches, parts_times, parts_delta_times,
                key, time_signature, fname, quarter_length,
                chords, chord_durations) = r

                all_transposed_chord.append(chords)
                all_transposed_chord_duration.append(chord_durations)
                all_transposed_pitch.append(pitches)
                all_transposed_parts_times.append(parts_times)
                all_transposed_parts_delta_times.append(parts_delta_times)
                all_transposed_keys.append(key)
                all_time_sigs.append(time_signature)
                all_file_names.append(fname)
                all_quarter_length.append(quarter_length)
            else:
                print("Result {} timed out".format(n))
        gtime = time.time()
        if verbose:
            r = gtime - itime
            print("Overall time {}".format(r))
        d = {"data_pitch": all_transposed_pitch,
             "data_parts_times": all_transposed_parts_times,
             "data_parts_delta_times": all_transposed_parts_delta_times,
             "data_key": all_transposed_keys,
             "data_time_sig": all_time_sigs,
             "data_chord": all_transposed_chord,
             "data_chord_duration": all_transposed_chord_duration,
             "data_quarter_length": all_quarter_length,
             "file_names": all_file_names}
        with open(pickle_path, "wb") as f:
            print("Saving pickle file %s" % pickle_path)
            pickle.dump(d, f)
        print("Pickle file %s saved" % pickle_path)
    else:
        print("Loading cached data from {}".format(pickle_path))
        with open(pickle_path, "rb") as f:
            d = pickle.load(f)

    major_pitch = []
    minor_pitch = []

    major_time_sigs = []
    minor_time_sigs = []

    major_part_times = []
    minor_part_times = []

    major_part_delta_times = []
    minor_part_delta_times = []

    major_chord = []
    minor_chord = []

    major_chord_duration = []
    minor_chord_duration = []

    major_filename = []
    minor_filename = []

    major_quarter_length = []
    minor_quarter_length = []

    major_part_times = []
    minor_part_times = []

    major_time_sigs = []
    minor_time_sigs = []

    keys = []
    for i in range(len(d["data_key"])):
        k = d["data_key"][i]
        ts = d["data_time_sig"][i]
        ddp = d["data_pitch"][i]
        ddt = d["data_parts_times"][i]
        ddtd = d["data_parts_delta_times"][i]
        nm = d["file_names"][i]
        ql = d["data_quarter_length"][i]
        try:
            ch = d["data_chord"][i]
            chd = d["data_chord_duration"][i]
        except IndexError:
            ch = "null"
            chd = -1

        if "major" in k:
            major_pitch.append(ddp)
            major_time_sigs.append(ts)
            major_part_times.append(ddt)
            major_part_delta_times.append(ddtd)
            major_filename.append(nm)
            major_chord.append(ch)
            major_chord_duration.append(chd)
            major_quarter_length.append(ql)
            keys.append(k)
        elif "minor" in k:
            minor_pitch.append(ddp)
            minor_time_sigs.append(ts)
            minor_part_times.append(ddt)
            minor_part_delta_times.append(ddtd)
            minor_filename.append(nm)
            minor_chord.append(ch)
            minor_chord_duration.append(chd)
            minor_quarter_length.append(ql)
            keys.append(k)
        else:
            raise ValueError("Unknown key %s" % k)

    all_pitches = major_pitch + minor_pitch
    all_time_sigs = major_time_sigs + minor_time_sigs
    all_part_times = major_part_times + minor_part_times
    all_part_delta_times = major_part_delta_times + minor_part_delta_times
    all_filenames = major_filename + minor_filename
    all_chord = major_chord + minor_chord
    all_chord_duration = major_chord_duration + minor_chord_duration
    all_quarter_length = major_quarter_length + minor_quarter_length

    all_notes = np.unique([ni for p in all_pitches for pi in p for ni in pi])
    n_notes = len(all_notes)

    final_chord_set = []
    final_chord_duration_set = []
    for n in range(len(all_chord)):
        final_chord_set.extend(all_chord[n])
        final_chord_duration_set.extend(all_chord_duration[n])

    final_chord_set = sorted(set(final_chord_set))
    final_chord_lookup = {k: v for k, v in zip(final_chord_set, range(len(final_chord_set)))}
    final_chord_duration_set = sorted(set(final_chord_duration_set))
    final_chord_duration_lookup = {k: v for k, v in zip(final_chord_duration_set, range(len(final_chord_duration_set)))}

    final_chord = []
    final_chord_duration = []
    for n in range(len(all_chord)):
        final_chord.append(np.array([final_chord_lookup[ch] for ch in all_chord[n]]).astype(floatX))
        final_chord_duration.append(np.array([final_chord_duration_lookup[chd] for chd in all_chord_duration[n]]).astype(floatX))

    final_pitches = []
    final_time_sigs = []
    final_durations = []
    final_part_times = []
    final_part_delta_times = []
    final_filenames = []
    final_keys = []
    final_quarter_length = []

    invalid_idx = []
    for i in range(len(all_pitches)):
        n = len(all_pitches[i])
        if lower_voice_limit is None and upper_voice_limit is None:
            cond = True
        else:
            raise ValueError("Voice limiting not yet implemented...")

        #if cond:
        if n == equal_voice_count:
            final_pitches.append(all_pitches[i])
            final_time_sigs.append(all_time_sigs[i])
            final_part_times.append(all_part_times[i])
            final_part_delta_times.append(all_part_delta_times[i])
            final_filenames.append(all_filenames[i])
            final_keys.append(keys[i])
            final_quarter_length.append(all_quarter_length[i])
        else:
            invalid_idx.append(i)
            if verbose:
                print("Skipping file {}: {} had invalid note count {}, {} required".format(
                    i, all_filenames[i], n, equal_voice_count))

    # drop and align
    final_chord = [fc for n, fc in enumerate(final_chord)
                   if n not in invalid_idx]
    final_chord_duration = [fcd for n, fcd in enumerate(final_chord_duration)
                            if n not in invalid_idx]

    all_chord = final_chord
    all_chord_duration = final_chord_duration
    all_time_sigs = final_time_sigs

    all_pitches = final_pitches
    all_part_times = final_part_times
    all_part_delta_times = final_part_delta_times
    all_filenames = final_filenames
    all_keys = final_keys
    all_quarter_length = final_quarter_length

    pitch_list = list(np.unique([ni for p in all_pitches for pi in p for ni in pi]))
    part_delta_times_list = list(np.unique([ni for pdt in all_part_delta_times for pdti in pdt for ni in pdti]))

    basic_durs = [.125, .25, .33, .5, .66, .75, 1., 1.5, 2., 2.5, 3, 3.5, 4., 5., 6., 8.]
    if len(part_delta_times_list) > len(basic_durs):
        from scipy.cluster.vq import kmeans2, vq
        raise ValueError("Duration clustering nyi")

        #cent, lbl = kmeans2(np.array(duration_list), 200)

        # relative to quarter length

        ul = np.percentile(duration_list, 90)
        duration_list = [dl if dl < ul else ul for dl in duration_list]
        counts, tt = np.histogram(duration_list, 30)
        cent = tt[:-1] + (tt[1:] - tt[:-1]) * .5
        cent = cent[cent > basic_durs[-1]]
        cent = sorted(basic_durs + list(cent))

        all_durations_new = []
        for adi in all_durations:
            shp = adi.shape
            fixed = vq(adi.flatten(), cent)[0]
            fixed = fixed.astype(floatX)

            code_where = []
            for n, ci in enumerate(cent):
                code_where.append(np.where(fixed == n))

            for n, cw in enumerate(code_where):
                fixed[cw] = cent[n]

            fixed = fixed.reshape(shp)
            all_durations_new.append(fixed)
        all_durations = all_durations_new
        duration_list = list(np.unique(np.concatenate([np.unique(adi) for adi in all_durations])))

    pitch_lu = {k: v for v, k  in enumerate(pitch_list)}
    duration_lu = {k: v for v, k in enumerate(part_delta_times_list)}

    quarter_length_list = sorted([float(ql) for ql in list(set(all_quarter_length))])
    all_quarter_length = [float(ql) for ql in all_quarter_length]

    r = {"list_of_data_pitch": all_pitches,
         "list_of_data_time": all_part_times,
         "list_of_data_time_delta": all_part_delta_times,
         "list_of_data_key": all_keys,
         "list_of_data_time_sig": all_time_sigs,
         "list_of_data_chord": all_chord,
         "list_of_data_chord_duration": all_chord_duration,
         "list_of_data_quarter_length": all_quarter_length,
         "chord_list": final_chord_set,
         "chord_duration_list": final_chord_duration_set,
         "pitch_list": pitch_list,
         "part_delta_times_list": part_delta_times_list,
         "quarter_length_list": quarter_length_list,
         "filename_list": all_filenames}
    return r


def check_fetch_bach_chorales_music21():
    """ Move files into pthbldr dir, in case python is on nfs. """
    from music21 import corpus
    all_bach_paths = corpus.getComposer("bach")
    partial_path = get_dataset_dir("bach_chorales_music21")
    for path in all_bach_paths:
        if "riemenschneider" in path:
            continue
        filename = os.path.split(path)[-1]
        local_path = os.path.join(partial_path, filename)
        if not os.path.exists(local_path):
            shutil.copy2(path, local_path)
    return partial_path


def fetch_bach_chorales_music21(keys=["B major", "D minor"],
                                truncate_length=100,
                                equal_voice_count=4,
                                force_denom=4,
                                compress_pitch=False,
                                compress_duration=False,
                                verbose=True):
    """
    Bach chorales, transposed to C major or A minor (depending on original key).
    Only contains chorales with 4 voices populated.
    Requires music21.

    n_timesteps : 34270
    n_features : 4
    n_classes : 12 (duration), 54 (pitch)

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["list_of_data_pitch"] : list of array
            Pitches for each piece
        summary["list_of_data_duration"] : list of array
            Durations for each piece
        summary["list_of_data_key"] : list of str
            String key for each piece
        summary["list_of_data_chord"] : list of str
            String chords for each piece
        summary["list_of_data_chord_duration"] : list of str
            String chords for each piece
        summary["pitch_list"] : list
        summary["duration_list"] : list

    pitch_list and duration_list give the mapping back from array value to
    actual data value.
    """

    data_path = check_fetch_bach_chorales_music21()
    pickle_path = os.path.join(data_path, "__processed_bach.pkl")
    mu = _music_extract(data_path, pickle_path, ext=".mxl",
                        skip_chords=False, equal_voice_count=equal_voice_count,
                        force_denom=force_denom,
                        verbose=verbose)

    lp = mu["list_of_data_pitch"]
    lt = mu["list_of_data_time"]
    ltd = mu["list_of_data_time_delta"]
    lql = mu["list_of_data_quarter_length"]

    del mu["list_of_data_chord"]
    del mu["list_of_data_chord_duration"]
    del mu["chord_list"]
    del mu["chord_duration_list"]

    def _len_prune(l):
        return [[lii[:truncate_length] for lii in li] for li in l]

    lp2 = _len_prune(lp)
    lt2 = _len_prune(lt)
    ltd2 = _len_prune(ltd)

    def _key_prune(l):
        k = mu["list_of_data_key"]
        assert len(l) == len(k)
        return [li for li, ki in zip(l, k) if ki in keys]

    lp2 = _key_prune(lp2)
    lt2 = _key_prune(lt2)
    ltd2 = _key_prune(ltd2)
    lql2 = _key_prune(lql)

    lp = lp2
    lt = lt2
    ltd = ltd2
    lql = lql2

    mu["list_of_data_pitch"] = lp
    mu["list_of_data_time"] = lt
    mu["list_of_data_time_delta"] = ltd
    mu["list_of_data_quarter_length"] = lql
    return mu


def quantized_to_pretty_midi(quantized,
                             quantized_bin_size,
                             save_dir="samples",
                             name_tag="sample_{}.mid",
                             add_to_name=0,
                             lower_pitch_limit=12,
                             list_of_quarter_length=None,
                             max_hold_bars=1,
                             default_quarter_length=47,
                             voice_params="woodwinds"):
    """
    takes in list of list of list, or list of array with axis 0 time, axis 1 voice_number (S,A,T,B)
    outer list is over samples, middle list is over voice, inner list is over time
    """

    is_seq_of_seq = False
    try:
        quantized[0][0]
        if not hasattr(quantized[0], "flatten"):
            is_seq_of_seq = True
    except:
        try:
            quantized[0].shape
        except AttributeError:
            raise ValueError("quantized must be a sequence of sequence (such as list of array, or list of list) or numpy array")

    # list of list or mb?
    n_samples = len(quantized)
    all_pitches = []
    all_durations = []

    max_hold = int(max_hold_bars / quantized_bin_size)
    if max_hold < max_hold_bars:
        max_hold = max_hold_bars

    for ss in range(n_samples):
        pitches = []
        durations = []
        if is_seq_of_seq:
            voices = len(quantized[ss])
            qq = quantized[ss]
        else:
            voices = quantized[ss].shape[1]
            qq = quantized[ss].T
        for i in range(voices):
            q = qq[i]
            pitch_i = [0]
            dur_i = []
            cur = None
            count = 0
            for qi in q:
                if qi != cur:# or count > max_hold:
                    if cur is None:
                        cur = qi
                        count += 1
                        continue
                    pitch_i.append(qi)
                    quarter_count = quantized_bin_size * (count + 1)
                    dur_i.append(quarter_count)
                    cur = qi
                    count = 0
                else:
                    count += 1
            quarter_count = quantized_bin_size * (count + 1)
            dur_i.append(quarter_count)
            pitches.append(pitch_i)
            durations.append(dur_i)
        all_pitches.append(pitches)
        all_durations.append(durations)
    pitches_and_durations_to_pretty_midi(all_pitches, all_durations,
                                         save_dir=save_dir,
                                         name_tag=name_tag,
                                         add_to_name=add_to_name,
                                         lower_pitch_limit=lower_pitch_limit,
                                         list_of_quarter_length=list_of_quarter_length,
                                         default_quarter_length=default_quarter_length,
                                         voice_params=voice_params)


def pitches_and_durations_to_pretty_midi(pitches, durations,
                                         save_dir="samples",
                                         name_tag="sample_{}.mid",
                                         add_to_name=0,
                                         lower_pitch_limit=12,
                                         list_of_quarter_length=None,
                                         default_quarter_length=47,
                                         voice_params="woodwinds"):
    # allow list of list of list
    """
    takes in list of list of list, or list of array with axis 0 time, axis 1 voice_number (S,A,T,B)
    outer list is over samples, middle list is over voice, inner list is over time
    durations assumed to be scaled to quarter lengths e.g. 1 is 1 quarter note
    2 is a half note, etc
    """
    is_seq_of_seq = False
    try:
        pitches[0][0]
        durations[0][0]
        if not hasattr(pitches, "flatten") and not hasattr(durations, "flatten"):
            is_seq_of_seq = True
    except:
        raise ValueError("pitches and durations must be a list of array, or list of list of list (time, voice, pitch/duration)")

    if is_seq_of_seq:
        if hasattr(pitches[0], "flatten"):
            # it's a list of array, convert to list of list of list
            pitches = [[[pitches[i][j, k] for j in range(pitches[i].shape[0])] for k in range(pitches[i].shape[1])] for i in range(len(pitches))]
            durations = [[[durations[i][j, k] for j in range(durations[i].shape[0])] for k in range(durations[i].shape[1])] for i in range(len(durations))]


    import pretty_midi
    # BTAS mapping
    def weird():
        voice_mappings = ["Sitar", "Orchestral Harp", "Acoustic Guitar (nylon)",
                          "Pan Flute"]
        voice_velocity = [20, 80, 80, 40]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., .95]
        return voice_mappings, voice_velocity, voice_offset, voice_decay

    if voice_params == "weird":
        voice_mappings, voice_velocity, voice_offset, voice_decay = weird()
    elif voice_params == "weird_r":
        voice_mappings, voice_velocity, voice_offset, voice_decay = weird()
        voice_mappings = voice_mappings[::-1]
        voice_velocity = voice_velocity[::-1]
        voice_offset = voice_offset[::-1]
    elif voice_params == "nylon":
        voice_mappings = ["Acoustic Guitar (nylon)"] * 4
        voice_velocity = [20, 16, 25, 10]
        voice_offset = [0, 0, 0, -12]
        voice_decay = [1., 1., 1., 1.]
        voice_decay = voice_decay[::-1]
    elif voice_params == "legend":
        # LoZ
        voice_mappings = ["Acoustic Guitar (nylon)"] * 3 + ["Pan Flute"]
        voice_velocity = [20, 16, 25, 5]
        voice_offset = [0, 0, 0, -12]
        voice_decay = [1., 1., 1., .95]
    elif voice_params == "organ":
        voice_mappings = ["Church Organ"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [.98, .98, .98, .98]
    elif voice_params == "piano":
        voice_mappings = ["Acoustic Grand Piano"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "electric_piano":
        voice_mappings = ["Electric Piano 1"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "harpsichord":
        voice_mappings = ["Harpsichord"] * 4
        voice_velocity = [40, 30, 30, 60]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    elif voice_params == "woodwinds":
        voice_mappings = ["Bassoon", "Clarinet", "English Horn", "Oboe"]
        voice_velocity = [50, 30, 30, 40]
        voice_offset = [0, 0, 0, 0]
        voice_decay = [1., 1., 1., 1.]
    else:
        # eventually add and define dictionary support here
        raise ValueError("Unknown voice mapping specified")

    # normalize
    mm = float(max(voice_velocity))
    mi = float(min(voice_velocity))
    dynamic_range = min(80, (mm - mi))
    # keep same scale just make it louder?
    voice_velocity = [int((80 - dynamic_range) + int(v - mi)) for v in voice_velocity]

    if not is_seq_of_seq:
        order = durations.shape[-1]
    else:
        try:
            # TODO: reorganize so list of array and list of list of list work
            order = durations[0].shape[-1]
        except:
            order = len(durations[0])
    voice_mappings = voice_mappings[-order:]
    voice_velocity = voice_velocity[-order:]
    voice_offset = voice_offset[-order:]
    voice_decay = voice_decay[-order:]
    if not is_seq_of_seq:
        pitches = [pitches[:, i, :] for i in range(pitches.shape[1])]
        durations = [durations[:, i, :] for i in range(durations.shape[1])]

    n_samples = len(durations)
    for ss in range(n_samples):
        durations_ss = durations[ss]
        pitches_ss = pitches[ss]
        # same number of voices
        assert len(durations_ss) == len(pitches_ss)
        # time length match
        assert all([len(durations_ss[i]) == len(pitches_ss[i]) for i in range(len(pitches_ss))])
        pm_obj = pretty_midi.PrettyMIDI()
        # Create an Instrument instance for a cello instrument
        def mkpm(name):
            return pretty_midi.instrument_name_to_program(name)

        def mki(p):
            return pretty_midi.Instrument(program=p)

        pm_programs = [mkpm(n) for n in voice_mappings]
        pm_instruments = [mki(p) for p in pm_programs]

        if list_of_quarter_length is None:
            # qpm to s per quarter = 60 s per min / quarters per min
            time_scale = 60. / default_quarter_length
        else:
            time_scale = 60. / list_of_quarter_length[ss]

        time_offset = np.zeros((order,))

        # swap so that SATB order becomes BTAS for voice matching
        pitches_ss = pitches_ss[::-1]
        durations_ss = durations_ss[::-1]

        # time
        for ii in range(len(durations_ss[0])):
            # voice
            for jj in range(order):
                try:
                    pitches_isj = pitches_ss[jj][ii]
                    durations_isj = durations_ss[jj][ii]
                except IndexError:
                    # voices may stop short
                    continue
                p = int(pitches_isj)
                d = durations_isj
                if d < 0:
                    continue
                if p < 0:
                    continue
                # hack out the whole last octave?
                s = time_scale * time_offset[jj]
                e = time_scale * (time_offset[jj] + voice_decay[jj] * d)
                time_offset[jj] += d
                if p < lower_pitch_limit:
                    continue
                note = pretty_midi.Note(velocity=voice_velocity[jj],
                                        pitch=p + voice_offset[jj],
                                        start=s, end=e)
                # Add it to our instrument
                pm_instruments[jj].notes.append(note)
        # Add the instrument to the PrettyMIDI object
        for pm_instrument in pm_instruments:
            pm_obj.instruments.append(pm_instrument)
        # Write out the MIDI data

        sv = save_dir + os.sep + name_tag.format(ss + add_to_name)
        try:
            pm_obj.write(sv)
        except ValueError:
            print("Unable to write file {} due to mido error".format(sv))
